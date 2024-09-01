# 🔭 Transfer Learning with Pytorch 
<div style="text-align: justify;">

<div style="border: 1px solid #ccc; padding: 10px;">
<b>This project is part of a series of assignments for COMP5329 - Deep Learning for the Masters Degree in Data Science at The University of Sydney. </b> (i) Snippets of codes are provided in this workbook <ins>for elaboration purposes</ins>. Please access the individual python files for full code. (ii) I've excluded the produced .pth files from this repo, they're taking too much space and I doubt they're any useful to anyone. (iii) Please go to the end of the document for attributions/citations. I've removed them from the text body for ease of reading.

<br> 

**Relevant links:**
- Github Repo can be accessed here  
- Train and validation scores is hosted on Tableau [here](https://public.tableau.com/app/profile/vincent.yunansan/viz/log_analyzer/Dashboard3?publish=yes)
- Colab workbook to train the best model can be found [here](https://colab.research.google.com/drive/1N4Q6oCltH2wJZLkh9Lo1Ya8VdE4WO9OR?usp=sharing)
<br>
</div>
<br>

**Table of Content:**
- [Project Background and Summary](#project-background-and-summary)
- [Exploratory Data Analysis and Pre-Processing](#exploratory-data-analysis-and-pre-processing)
- [Design Decisions](#design-decisions)
- [Additional code snippets](#additional-code-snippets-non-exhaustive)
- [Models used in Transfer learning](#models-used-for-transfer-learning)
- [Example on How to Run a Training Cycle](#example-on-how-to-run-a-training-cycle)
- [Results on Transfer Learning](#result--discussion-of-experiment-1-transfer-learning)
- [Results on Mixed Precision Training](#result--discussion-of-experiment-2-mixed-precision)
- [Final Words](#final-words)
- [References](#references)

## Project Background and Summary
The goal of this project is to implement a multi-label classification model on 30,000 images split into 18 classes. The target model is required to (i) achieve test micro-F1 score of 85%+, (ii) have size under 100mb, and (iii) require less than  24 GPU-hours of training time. 

See sample images, labels, and captions below:

<p align="center">
  <img src="assets/sampleimages.png" height= "250">
  <br>
  <b>Sample data points from the train set. Notice some images have multiple labels.</b>
</p>

To efficiently solve this problem, we utilized `transfer learning`, where several pre-trained models available on pytorch were fine-tuned to our task at hand. These models were chosen for their size (<100mb) and known performance (they were trained on ImageNet, which is a much larger computer vision problem set compared to our project). These models were fine-tuned on various optimizers and parameter values to identify the best model configuration.

<p align="center">
  <img src="assets/modellist.png" height ="150">
  <br>
  <b>List of pre-trained models explored.</b>
</p>

We then propose the use of mixed precision training to the final model which will allow for faster training at the cost of lower scores. 

## Exploratory data analysis and pre-processing

`Exploratory data analysis`: The dataset contains images of different sizes and one label is missing. We used the NLTK library along with casual observation to identify the general meaning of each label. We note the significant data imbalance and co-occurence amongts several labels.

<p align="center">
  <img src="assets/names.png" height ="350">
  <br>
  <b>Names assigned based on their frequency of occurence in captions (excluding stop words) and heuristics. Some labels shows up together with other labels more often than others.</b>
</p>


`Pre-processing`: We applied several pre-processing strategies to add noise to the dataset, which will help with model robustness (at the cost of training time). We then feed the the transformer to the DataLoader for use with Pytorch.


<p align="center">
  <img src="assets/transformation.png" height ="200">
  <br>
  <b>Transformations include resizing, random cropping, color jitter, rotation, flips, grey scalling. The transformed images were then transformed into tensors, normalized, and loaded to the Pytorch DataLoader.</b>
</p>

<div style="font-size: 9px;">

```python
#apply transformation to image(s)
transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness = 0.5),
    transforms.RandomRotation(degrees = 45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#define the Image class
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx, 1:].values.astype('float32') 
        
        return image, label 

#Load input images to the class and then to dataloader.
train_dataset = ImageDataset(train_dataset, image_dir, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=None)
```
</div>


## Design decisions

- `Our model is trained to optimize for macro-F1`: we treat all classes equally, providing a balanced view of the model’s performance across all classes. We would like to  ensure that the model performs well across all classes, regardless of class size. Per our observation, (validation) macro-average F1 score of 70.0%+ generally translates to (test) micro-average F1 score of 88.5%+. 

| | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: |
| <b>Micro F1</b> is used in test as specified in the project description | $$P_{\text{micro}} = \frac{\sum_{i=1}^{n} \text{TP}_i}{\sum_{i=1}^{n} (\text{TP}_i + \text{FP}_i)}$$ | $$R_{\text{micro}} = \frac{\sum_{i=1}^{n} \text{TP}_i}{\sum_{i=1}^{n} (\text{TP}_i + \text{FN}_i)}$$ | $$F1_{\text{micro}} = \frac{2 \cdot P_{\text{micro}} \cdot R_{\text{micro}}}{P_{\text{micro}} + R_{\text{micro}}}$$ |
| <b>Macro F1</b> is used in training to emphasize the importance of balanced output | $$P_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$$ | $$R_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$$ | $$F1_{\text{macro}} = \frac{1}{n} \sum_{i=1}^{n}\frac{2 \cdot P_i \cdot R_i}{P_i + R_i}$$ |

- `We decided to ignore captions in training our model to simulate real world situation`: The use/dependency on available caption creates a Catch-22 situation. A high quality caption depends on visual analysis of the image and accurate human annotation. A visual classification model that depends on caption input will therefore be unnecessary in the first place.

- `We use transfer learning for efficiency`: This project utilizes pre-trained on ImageNet as the starting point for models used to solve our task. This technique allows for certainty of usability (e.g. we know the models work on much larger image classification problems) and significant savings in training resources (e.g. we do not need to train the model from scratch).

  All pre-trained models used are less than 100 megabytes to adhere with the project specification and are part of an open-source library (PyTorch). To further accelerate training time, the team also implemented dynamic learning rate, early stopping, mini-batch training, and a range of criterions to fine the best mix which yields the best outcome. The wide range of models and optimizers are intended to dentify the ‘sweet spot’ of strong performance.

   `Binary Cross Entropy with Logits (BCE with logits)` was used as criterion for this multi-label classification problem.  BCE with logits is two-step calculation where probability of a label is calculated and binarized through a sigmoid hurdle. The final layer of our models produces an output matrix which has width equivalent to the number of classes (e.g. 18 classes ) and each cell in the output matrix yields 1 (True) for any classes that are predicted to be in the image and 0 (False) otherwise.

- `Mixed precision training on the best model`: The team then suggests one best model and mixed precision training to analyse the trade-off between saved computational cost and lost performance. Other improvement ideas that are not implemented in this project are discussed in the last chapter.

## Additional code snippets (non-exhaustive)

- `label one hot encoding`: we transformed the label string to binary vectors for multi-label classification.

<div style="font-size: 9px;">

```python
def one_hot_encode_multilabels(df, label_column):

    # Split the labels by space and apply one-hot encoding, convert to int
    s = df[label_column].str.split().explode()
    s = s.astype(int)

    # Get dummies and then sum back to original DataFrame shape
    dummies = pd.get_dummies(s, prefix='Label').groupby(level=0).sum()

    # Concatenate the original DataFrame with the new one-hot encoded columns
    df_encoded = pd.concat([df, dummies], axis=1)

    return df_encoded
```
</div>

- `Logger`: we recorded loss scores & training times for analysis purposes. This handler both print out and saves the training logs.

<div style="font-size: 9px;">

  ```python
  class Logger:
    def __init__(self, model_name, timestamp, logging_path='models/logs/'):
        self.model_name = model_name
        self.timestamp = timestamp
        self.logging_path = logging_path
        os.makedirs(self.logging_path, exist_ok=True)

        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.handlers = []  # Clear existing handlers, otherwise we'll get multiple as has happened
        self.logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'{self.logging_path}{self.model_name}.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def get_logger(self):
        return self.logger

  ```
</div>

- `Early stopping`: training is stopped after validation score doesn't improve after a certain number of cycles (default:4 cycles)

<div style="font-size: 9px;">

  ```python
  class EarlyStopper:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
  ```

</div>

- `Generate prediction`: this function generates prediction output and saves it as a csv file. Prediction for each class is implemented with a sigmoid hurdle as shown below.

<div style="font-size: 9px;">

```python
def generate_predictions_and_save(model,test_data, output_name, output_path = 'models/output/'):
  
    #we don't explicitly call model in this function
    model.eval()
    model.to(device)
    
    # Generate prediction
    test_predictions = []
    test_image_ids = test_data['ImageID'].tolist()

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).data > 0.5
            test_predictions.extend(predictions.cpu().numpy())

    # Get label names from df_encoded_train
    label_columns = [col for col in df_encoded_train.columns if col.startswith('Label_')]
    label_names = [col.split('_')[1] for col in label_columns]  

    # Convert predictions to indices
    predicted_labels = [' '.join(label_names[j] for j in range(len(label_names)) if pred[j]) for pred in test_predictions]

    # Combine ImageID and predicted Labels into a DataFrame
    results_df = pd.DataFrame({
        'ImageID': test_image_ids,
        'Labels': predicted_labels
    })

    output_file = f'{output_path}{output_name}.csv'

    # Save as csv
    results_df.to_csv(output_file, index=False)
```

</div>

- `Train and val run`: This (long) code block initializes the model, optimizer, number of epochs and other optional arguments. It utilizes the learning rate scheduler which reduces the learning rate when the loss stops improving. It utilizes BCEwithLogitsLoss as explained previously. The function logs all relevant training and validation metrics to a file for further analysis. Once the model has finished training (e.g. when early stopping is triggered or the loop reaches maximum epoch), a final model is saved.

<div style="font-size: 9px;">

```python
def train_val_run(model, optimizer, output_name, test_data, epochs, validate_every = 2, lr_scheduler_toggle = True):
    
    #extract determined model name and pre-trained model, assign to MPS/CUDA
    model.to(device)

    #setup scheduler
    if lr_scheduler_toggle:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    #set criterion 
    criterion = torch.nn.BCEWithLogitsLoss()

    #setup loggers
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    logger = Logger(output_name,timestamp).get_logger()
        
    #record runtimes for stats
    run_time = []

    #implement early stopper to prevent overfitting
    early_stopper = EarlyStopper(patience=6, min_delta=0.001)

    #training and validation loop

    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr'] 
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        run_time.append(epoch_time)
        logger.info(f'Training   >>> Epoch: {epoch+1}, Current LR: {current_lr}, Loss: {(running_loss/len(train_loader)):.3f}, Time: {int(epoch_time // 60)}m{int(epoch_time % 60)}s')

        # Validation
        if (epoch + 1) % validate_every == 0 or (epoch + 1) == epochs:
            model.eval()
            validation_loss = 0.0
            all_labels = []
            all_predictions = []
            start_time = time.time()

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss = criterion(outputs, labels)
                    validation_loss += val_loss.item()
                    predictions = torch.sigmoid(outputs).data > 0.5  
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())

            #calculate validation loss and update scheduler
            validation_loss /= len(val_loader)
            val_time = time.time() - start_time
            run_time.append(val_time)

            #calculate f1 score
            all_labels = np.vstack(all_labels)
            all_predictions = np.vstack(all_predictions)
            f1_scores = f1_score(all_labels, all_predictions, average='macro')

            logger.info(f'Validation >>> Epoch: {epoch+1}, Loss: {(validation_loss):.3f}, current lr: {current_lr}, F1 Score: {f1_scores:.3f}, Time: {int(val_time // 60)}m{int(val_time % 60)}s')

            #updat scheduler
            if lr_scheduler_toggle:
                scheduler.step(validation_loss)

            #early stopper
            if early_stopper.early_stop(validation_loss):
                logger.info(f'Early stopping >>> triggered at epoch {epoch + 1}')
                break

    #total time
    total_time = sum(run_time)
    logger.info(f'Training concluded >>> Total Run Time: {int(total_time // 60)}m{int(total_time % 60)}s')

    #save model and show size

    # Create directory if it doesn't exist
    os.makedirs('models/pth', exist_ok=True)
    path = f'models/pth/{output_name}.pth'
    torch.save(model.state_dict(), path)
    file_size = os.path.getsize(path)
    logger.info(f"Model size on disk: {file_size / (1024 * 1024):.2f} MB")

    #generate prediction output
    generate_predictions_and_save(model,test_data, output_name)
```
</div>

- `Experiment wrapper`: This code block runs the experiments based on the input model, number of epochs, list of optmizers, and validation frequency.

<div style="font-size: 9px;">

```python
def run_experiments(model_name, model, epochs, optimizers, validate_every = 2):
    for optimizer in optimizers:
        optimizer_name = optimizer[0]
        optimizer_starting_lr = optimizer[1]
        optimizer_lr_scheduler_toggle = optimizer[2]
        
        if optimizer_name == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=optimizer_starting_lr)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_starting_lr) 
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_starting_lr)

        output_name = f'{model_name}_{optimizer_name}_{optimizer_starting_lr}'
        model = train_val_run(model, optimizer, output_name, test_data, epochs, lr_scheduler_toggle = optimizer_lr_scheduler_toggle)
```
</div>

## Models used for transfer learning

**Models used are as follows:**

- `GoogleNet (2014)`: developed by Google, this architecture is able to achieve significantly more depth (22 layers) compared to its predecessors at manageable computational costs by introducing (i) ‘inception modules’ which consists of 1x1, 3x3, and 5x5 convolutional filters and 3x3 pooling operations. Inception modules allow the network to ‘learn and choose’ from different sized filter in each layer, (ii) ‘auxiliary classifiers’ which are attached to intermediate layers. Auxiliary classifiers contribute weighted loss values from intermediate layers to the final loss calculation to address the vanishing gradient problem.

<p align="center">
  <img src="assets/googlenet.png" height ="150">
  <br>
  Inception module and auxiliary classifiers in GoogleNet
</p>

  - `ResNet (2015)`: introduced the use of ‘residual blocks’ and ‘shortcut connections’. Shortcut connections address the vanishing gradient problem by allowing input of an earlier layer to ‘skip’ one or more layers to become input of a deeper layer. This technique allows for much deeper networks, reaching 152 layers for ImageNet.

<p align="center">
  <img src="assets/resnet.png" height ="150">
  <br>
  Residual blocks and skip conections in ResNet
</p>

- `ResNext (2016)`: is an extension to ResNet which introduces additional branches called ‘cardinality’. Cardinality splits layers into multiple pathways, allowing models to scale up in a new dimension outside of depth and width, and aggregates the output at the end of the block. This technique is useful in expanding a model while maintaining complexity as model growth is ‘spread out’ to other dimensions.

<p align="center">
  <img src="assets/resnext.png" height ="150">
  <br>
  Cardinality in ResNext
</p>

- `ShuffleNet (2016)`: first developed in 2017 for “mobile devices with very limited computing power”, ShuffleNet utilizes (i) ‘group convolutions’ which convolves the input to reduce the number of parameters and (ii) ‘channel shuffle’ which allow group convolutions in the next layer to receive input from all preceding groups. ShuffleNetV2 is an iteration of ShuffleNet which introduces channel splitting and simplified design.

<p align="center">
  <img src="assets/shufflenet.png" height ="500">
  <br>
  Group convolutions in ShuffleNet
</p>

- `EfficientNet (2020)`: developed by Google Research, EfficientNet is a family of models developed under the idea of a ‘compound coefficient’ which systemically scales depth/width/resolution. The main building block of EfficientNet is the ‘mobile inverted bottleneck’ where (i) input is expanded, (ii) passed through depth-wise convolution to create non-linearity, and (iii) projected back to a lower dimension . 

<p align="center">
  <img src="assets/efficientnet.png" height ="200">
  <br>
  Compound coefficient in EfficientNet
</p>

- `RegNet (2020)`: developed by Facebook AI Research (FAIR), RegNet is based on the idea that model design parameters should be “simple, work well, and generalize across settings”. Experiments run by FAIR displayed clear quantized linear relationship between network parameters, such as depth and width. The authors of RegNet claims that “Regnet is able to achieve good results with simple models” and RegNet is able to “outperform EfficientNet by 5x on GPUs”. 

## Example on how to run a training cycle

The pre-trained model can be accessed on [Pytorch](https://pytorch.org/vision/main/models.html).   
Below is an example how a training run can be initiated using the provided helper codes. 

<div style="font-size: 9px;">

```python
model_name = 'resnext50'

# Create DataLoaders for both train and validation sets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#run train and val
for optimizer in optimizers:
    optimizer_name = optimizer[0]
    optimizer_starting_lr = optimizer[1]
    optimizer_lr_scheduler_toggle = optimizer[2]
    
    #reset model
    model = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.DEFAULT)
    model.num_classes = 18
    model.fc = torch.nn.Linear(model.fc.in_features, model.num_classes)  
    
    #call optimizer and starting learning rate:
    if optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=optimizer_starting_lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_starting_lr) 
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_starting_lr)

    output_name = f'{model_name}_{optimizer_name}_{optimizer_starting_lr}'
    train_val_run(model, optimizer, output_name, test_data, epochs, lr_scheduler_toggle = optimizer_lr_scheduler_toggle)
```
</div>

## Result & Discussion of Experiment 1 (Transfer Learning)

**Results**:

Hyperparameter testing is done on the 6 pre-trained models and 3 optimizers described above. The maximum number of epochs is set at 60 epochs to identify which models can converge quicker and provide better results within acceptable computation times. Logs, prediction output, and saved model are automatically generated and saved. Logs include measures such as epoch count, training time of the epoch, training and validation loss scores, and F1-score at validation. Results below:

<p align="center">
  <img src="assets/result11.png" height ="300">
  <br>
  Group convolutions in ShuffleNet
</p>

<p align="center">
  <img src="assets/result12.png" height ="500">
  <br>
  Group convolutions in ShuffleNet
</p>

**Models**:
- GoogleNet and ShuffleNet (6.6 and 7.4 million parameters each) underperform ResNet, ResNext, and RegNet (25.6, 25.0, 15.3 million parameters each). We hypothesize that GoogleNet and ShuffleNet are not converging fast enough due to their more simple architectures and will require higher epoch counts to get to equitable results.

- EfficientNet (82.7 million) failed to converge under multiple training attempts and was abandoned. We observed significant instability where a single epoch may run between 8 minutes to 500 minutes per epoch or not converging at all. Although EfficientNet is not the largest model by number of parameters, it is much heavier in terms of computational load at 8.37 GFLOPS, almost twice as heavy as the next model in our collection.

**Optimizers**:
- Optimizer performance: Adam is better than AdaDelta and SGD in under-performing models (e.g. GoogleNet and ShuffleNet) and overfits in stronger models (e.g. ResNet, ResNext, and RegNet). This suggests a ‘sweet spot’ between model vs. optimizer strengths.

**Test F1 Scores**:
- Validation and Test F1 Scores: ResNet, RegNet, and ResNext achieved close to or above 0.700 of (validation) macro-f1 score and were submitted to the Kaggle platform. (Test) micro-F1 score were similar at around 0.900.

**Best performers**: 

models produced by ResNet, ResNext, and RegNet passed our performance hurdle and achieved similar (test) micro-average F1 scores with the following highlights: (i) ResNet provided the best ‘bang for buck’ in terms of performance for training time, (ii) RegNet was best in terms of performance for size. Both ResNet and RegNet models were trained well within the 24 GPU-hour limit. As such, we defined RegNet trained with AdaDelta as our best model. See results below:

**Understanding RegNet model works**: 

RegNet is a deep learning network that is built upon ResNet. The architecture includes skip connections and regulatory modules using Convolutional Recurrent Neural Networks (‘Conv RNNs’). Just like the typical RNNs in deep learning, Conv RNNs utilize a hidden state to serve as a checkpoint to save spatiotemporal information of the image. This hidden state is constantly updated and used in the subsequent convolutions, which retains information in deeper convolutional layers. 

To showcase the inner functions of our RegNet model, we fed forward a single example through our RegNet model and visualized feature maps at different layers of the model (see Figure 16) and visualized a heatmap with grad-cam to highlight which feature areas are processed by the model in coming up with a classification prediction (see Figure 17).

By observing the feature map in Figure 16, we can see low-level features extracted in earlier layers becoming more complex/abstract as the layers went deeper. The abstraction in deeper  layers is due to sequential activations between layers, down-sampling (which lowered spatial resolution), and skip connections between alternate layers (which passes complexity to deeper layers to solve the vanishing gradient problem). This abstraction represents learning in a deep learning model.

The heatmap in Figure 17 shows the model identifying regions of the image that carries relevant information for class identification. Specific to the cat image used for this analysis, we observe how the model is able to differentiate the cat’s facial feature (high heatmap value, in red) against its body (neutral heatmap value, in green) and backdrop (lower heatmap value, in teal).


<p align="center">
  <img src="assets/regnetfeaturemap.png" height ="500">
  <br>
  Group convolutions in ShuffleNet
</p>
  
## Result & Discussion of Experiment 2 (Mixed Precision)

<div style="border: 1px solid #ccc; padding: 10px;">
  **NOTE**: Mixed Precision Training is a feature available on cuda GPU. Please set your device to `cuda` to run this experiment. 
</div>

<br> We tested the best model from Experiment 1 (RegNet with AdaDelta) on mixed precision training to explore the training time vs. performance trade-off. Mixed precision resulted in c.60% training time savings (at 60 epochs) with only 3 basis points of deterioration to the (validation) macro-average F1 score. Further training increased the (validation) macro-average F1 well beyond the previous best and (test) micro-average F1 scores are comparable (0.895 vs. 0.900).</br>

`Best model`: We define RegNet with AdaDelta and Mixed Precision as our best model as it satisfies our technical criteria in a significantly shorter amount of time.

<p align="center">
  <img src="assets/mixprec.png" height ="500">
  <br>
  <b>Group convolutions in ShuffleNet</b>
</p>

## Final words

This study highlights the potency of transfer learning whereby pre-trained models can be fine-tuned for smaller tasks, saving computational resources and time for the user. 

`Other development ideas not (yet) explored` are listed below. We did not employ these strategies as our project and stretch goals were already achived with our base protocol:
- Cross-validation can be implemented to increase model robustness and reliability. In simple terms, cross validation takes different cuts of the dataset, and do multiple training runs to come up with an average score. Cross validation provides a more reliable estimate of performance compared to a single train-test split.

- We can also utilize the caption data and create an ensemble model which takes both image and text data. The two streams of processes can be done in parallel and combined at the end to create a prediction output. 

- We can train a small model which takes the output of the larger pre-trained model out-of-the-box and predict classes as per our project description. The smaller boosting model can be connected to the larger pre-trained model on the FC layer with suitable transformations (e.g. batch normalization). This allows us to focus training on the smaller model and not fine-tune the pre-trained model.

## References

<div style="font-size: 9px;">

- Brownlee, J. (2019, August 19). Machine Learning Mastery. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/

- Chauhan, N. K., & K, S. (2018). A Review on Conventional Machine Learning vs Deep Learning. 2018 International Conference on Computing, Power and COmmunication Technologies (GUCON). 

- Chen, K., Bao-Liang, L., & Kwok, J. T. (2006). Efficient Classification of Multi-Label and Imbalanced Data using Min-Max Modular Classifiers. The 2006 IEEE International Joint Conference on Neural Network Proceedings. 

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv, 1512.03385.

- Huang, Y., Wang, W., Wang, L., & Tan, T. (2013). Multi-task deep neural network for multi-label learning. 2013 IEEE International Conference on Image Processing. 

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. Nature, 436-444.

- Leung, K. (2022, January 5). Towards Data Science. Retrieved from Towards Data Science: https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f

- Liu, W., Wang, W., Wang, L., & Tan, T. (2022). The Emerging Trends of Multi-Label Learning. IEEE Transactions oon Pattern Analysis and Machine Intelligence.

- Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018). ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. arXiv, 1807.11164v1.
- Marques, J. A., Gois, F. N., Madeiro, J. P., Li, T., & Fong, S. J. (2022). Chapter 4 - Artificial neural network-based approaches for computer-aided disease diagnosis and treatment. Cognitive and Soft Computing Techniques for the Analysis of Healthcare Data, 79-99.
- Min-Ling, Z., & Zhi-Hua, Z. (2006). Multilabel Neural Networks with Applications to functional Genomics and Text Categorization. IEEE Transactions on Knowledge and Data Engineering, 18(10), 1338-1351.
- Mohammed, A., & Kora, R. (2023). A comprehensive review on ensemble deep learning: Opportunities and challenges. Journal of King Saud University - COmputer and Information Sciences. Volume 35, Issue 2, 757-774.

- Multi-label Classification Competition 2024. (2024). Retrieved from Kaggle: https://kaggle.com/competitions/multi-label-classification-competition-2024

- PyTorch Contributors. (2023). Automatic Mixed Precision package - torch.amp . Retrieved from Pytorch: https://pytorch.org/docs/stable/amp.html

- PyTorch Contributors. (2023). Pytorch.org. Retrieved from Pytorch.org: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

- Radosavovic, I., Kosaraju, R., Girshick, R., He, K., & Dollar, P. (2020). Designing Network Design Spaces. arXiv, 2003.13678v1.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. IEEE/CVF Conference on Computer Vision and Pattern Recognition.

- Sharma, S., & Guleria, K. (2022). Deep learning Models for Image Classification: Comparison and Applications. 2022 2nd International Conference on Advance Computing and Innovative Technologies in Engineering(ICACITE).

- Szegedy, C., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., . . . Rabinovich, A. (2014). Going Deeper with Convolutions. arXiv, 1409.4842.
- Tan, M., & Le, Q. V. (2020). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv, 1905.1194v5.
- Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. arXiv, 2104.00298v3.
- Tao, L. (2024). Kaggle. Retrieved from Kaggle: https://www.kaggle.com/competitions/multi-label-classification-competition-2024
- Tidake, V. S., & Sane, S. S. (2018). Evaluation of Multi-label Classifiers in Various Domains Using Decision Tree. Intelligent Computing and Information and Communication.
- Wei, D. (2024, April 5). Medium. Retrieved from Medium: https://medium.com/@weidagang/demystifying-the-adam-optimizer-in-machine-learning-4401d162cb9e#:~:text=Here's%20why%20Adam%20has%20become,Stochastic%20Gradient%20Descent%20(SGD).
- Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). Aggregated Residual Transformations for Deep Neural Networks. arXiv, 1611.05431.
- Xu, J., YuPan, Pan, X., Hoi, S., Yi, Z., & Xu, Z. (2022). RegNet: Self-Regulated Network for Image Classification. IEEE Transactions on Neural Networks and Learning Systems. 
- Zhang, X., Zhou, X., Lin, M., & Sun, J. (2017). ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. arXiv, 1707.01083v2.












