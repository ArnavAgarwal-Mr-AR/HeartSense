library(data.table)
library(caTools)
library(ggplot2)
library(reshape2)
library(caret)
library(Boruta)
library(rpart)
library(rpart.plot)
library(knitr)
library(RColorBrewer)
library(gbm)
library(class)
library(e1071)
library(randomForest)
library(xgboost)
mydata <- read.csv("E:/SEM 6/EDA/Project/eda_heart.csv")
head(mydata)
dim(mydata)
str(mydata)
summary(mydata)
colSums(is.na(mydata))
mydata <- unique(mydata)
dim(mydata)

#Outliers
variables <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
boxplot_data <- mydata[, variables]

# Define the colors using a color palette from RColorBrewer
box_colors <- brewer.pal(length(variables), "Set2")

# Create the boxplot with customized colors
boxplot(boxplot_data, col = box_colors, main = "Boxplot of Continouos Variables")

# IQR methods to remove outliers
# Copy df for modification
mydata_clean <- mydata

# Iterate over continuous data columns only
for (col in c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")) {
  
  # Calculate Q1, Q3 & IQR
  q1 <- quantile(mydata[[col]], 0.25)
  q3 <- quantile(mydata[[col]], 0.75)
  iqr <- q3 - q1
  
  # Define the lower bound and upper bound
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  cat(col, ": lower bound is", round(lower_bound, 3), ", upper bound is", round(upper_bound, 3), "\n")
  
  # Remove outliers by filtering based on lower & upper bounds
  mydata_clean <- mydata_clean[mydata_clean[[col]] >= lower_bound & mydata_clean[[col]] <= upper_bound, ]
}
cat("Number of rows after remove outliers:", nrow(mydata_clean), "\n")
mydata <- mydata_clean

#Label encoding
unique(mydata$Sex)
mydata$Sex <- ifelse(mydata$Sex == "M",1,0)
# ChestPainType
unique(mydata$ChestPainType)
mydata$ChestPainType <- as.numeric(factor(mydata$ChestPainType,
                                          levels = c('ASY','ATA','NAP','TA' ),
                                          labels = c(1,2,3,4)))
# RestingECG
unique(mydata$RestingECG)
mydata$RestingECG <- as.numeric(factor(mydata$RestingECG,
                                       levels = c('Normal','ST','LVH'),
                                       labels = c(1,2,3)))
# ExerciseAngina
unique(mydata$ExerciseAngina)
mydata$ExerciseAngina <- ifelse(mydata$ExerciseAngina == "Y",1,0)
# ST_Slope
unique(mydata$ST_Slope)
mydata$ST_Slope <- as.numeric(factor(mydata$ST_Slope,
                                     levels = c('Up','Flat','Down'),
                                     labels = c(1,2,3)))
str(mydata)

#feature selection
feature_select <- Boruta(HeartDisease ~ ., data = mydata)
feature_select$finalDecision

#Data splitting
set.seed(123)
split <- sample.split(mydata$HeartDisease, SplitRatio = 0.7)
train_set <- subset(mydata, split == TRUE)
test_set <- subset(mydata, split == FALSE)
dim(train_set)
dim(test_set)

#Data transformation
train_set[, c(1, 4, 5, 8, 10)] = scale(train_set[, c(1, 4, 5, 8, 10)])
test_set[, c(1, 4, 5, 8, 10)] = scale(test_set[, c(1, 4, 5, 8, 10)])
head(train_set)
head(test_set)

#Univariate Analysis
#Histogram
variables <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
# Define the colors using a color palette from RColorBrewer
hist_colors <- brewer.pal(length(variables), "Set2")

for (i in seq_along(variables)) {
  col <- variables[i]
  if (is.numeric(train_set[[col]])) {
    hist(train_set[[col]], main = col, xlab = "", col = hist_colors[i])
  }
}
#Bivariate Analysis
#Correlation & Heatmap
# to see relationship between 2 continuous variables
pairs(train_set[c("Age", "RestingBP", "Cholesterol", "MaxHR")], main = "Relationship between Continous variable",
      col = "blue", pch = 20)
#create correlation heatmap
corr_matrix <- cor(train_set)

# Convert the correlation matrix to a data frame
corr_df <- melt(corr_matrix)

ggplot(data = corr_df, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = round(value, 2)), size = 3) +
  scale_fill_gradient2(low = "blue", high = "red",
                       limit = c(-1,1), name="Correlation") +
  theme_minimal() +
  labs(x = "", y = "", title = "Correlation Matrix") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels
#Bar charts
# Sex with heartdisease 
ggplot(train_set, aes(x = factor(Sex, labels = c("F", "M")))) +
  geom_bar(aes(fill = factor(HeartDisease, labels = c("N", "Y"))), position = "dodge") +
  labs(x = "Sex", fill = "Heart Disease")
# ChestPainType with heartdisease 
ggplot(train_set, aes(x = factor(ChestPainType, labels = c("ASY", "ATA", "NAP", "TA")))) +
  geom_bar(aes(fill = factor(HeartDisease, labels = c("N", "Y"))), position = "dodge") +
  labs(x = "ChestPainType", fill = "Heart Disease")
# ExerciseAngina with heartdisease 
ggplot(train_set, aes(x = factor(ExerciseAngina, labels = c("N", "Y")))) +
  geom_bar(aes(fill = factor(HeartDisease, labels = c("N", "Y"))), position = "dodge") +
  labs(x = "ExerciseAngina", fill = "Heart Disease")
# RestingECG with heartdisease 
ggplot(train_set, aes(x = factor(RestingECG, labels = c("Normal", "ST", "LVH")))) +
  geom_bar(aes(fill = factor(HeartDisease, labels = c("N", "Y"))), position = "dodge") +
  labs(x = "RestingECG", fill = "Heart Disease")
# ST_Slope with heartdisease 
ggplot(train_set, aes(x = factor(ST_Slope, labels = c("Up", "Flat", "Down")))) +
  geom_bar(aes(fill = factor(HeartDisease, labels = c("N", "Y"))), position = "dodge") +
  labs(x = "ST_Slope", fill = "Heart Disease")
# FastingBS with heartdisease 
ggplot(train_set, aes(x = factor(FastingBS, labels = c("Otherwise", "FastingBS > 120 mg/dl")))) +
  geom_bar(aes(fill = factor(HeartDisease, labels = c("N", "Y"))), position = "dodge") +
  labs(x = "FastingBS", fill = "Heart Disease")

#Multivariate analysis
#Scatterplots
# Age & RestingBP with heartdisease
ggplot(train_set, aes(x = Age, y = RestingBP, shape = factor(HeartDisease))) +
  geom_point(aes(color = factor(HeartDisease))) +
  labs(x = "Age", y = "RestingBP") +
  scale_shape_manual(values = c(19, 15)) +
  theme_classic() 
# Age & Cholesterol with heartdisease 
ggplot(train_set, aes(x = Age, y = Cholesterol, shape = factor(HeartDisease))) +
  geom_point(aes(color = factor(HeartDisease))) +
  labs(x = "Age", y = "Cholesterol") +
  scale_shape_manual(values = c(19, 15)) +
  theme_classic() 
# Age & MaxHR with heartdisease 
ggplot(train_set, aes(x = Age, y = MaxHR, shape = factor(HeartDisease))) +
  geom_point(aes(color = factor(HeartDisease))) +
  labs(x = "Age", y = "MaxHR") +
  scale_shape_manual(values = c(19, 15)) +
  theme_classic()      
# RestingBP & Cholesterol with heartdisease 
ggplot(train_set, aes(x = RestingBP, y = Cholesterol, shape = factor(HeartDisease))) +
  geom_point(aes(color = factor(HeartDisease))) +
  labs(x = "RestingBP", y = "Cholesterol") +
  scale_shape_manual(values = c(19, 15)) +
  theme_classic()     
# RestingBP & MaxHR with heartdisease 
ggplot(train_set, aes(x = RestingBP, y = MaxHR, shape = factor(HeartDisease))) +
  geom_point(aes(color = factor(HeartDisease))) +
  labs(x = "RestingBP", y = "MaxHR") +
  scale_shape_manual(values = c(19, 15)) +
  theme_classic()  
# Cholesterol & MaxHR with heartdisease 
ggplot(train_set, aes(x = Cholesterol, y = MaxHR, shape = factor(HeartDisease))) +
  geom_point(aes(color = factor(HeartDisease))) +
  labs(x = "Cholestrol", y = "MaxHR") +
  scale_shape_manual(values = c(19, 15)) +
  theme_classic()  

#Data modelling
#Logistic regression
# Fit logistic regression model
log_model <- glm(HeartDisease ~ ., data = train_set, family = binomial())

# Make predictions on test set
log_pred <- predict(log_model, newdata = test_set, type = "response")

# Create confusion matrix
log_cm <- table(test_set$HeartDisease, log_pred > 0.5)

# Calculate accuracy
log_accuracy <- sum(diag(log_cm))/sum(log_cm)

# Compute other evaluation metrics
log_precision <- log_cm[2,2] / sum(log_cm[,2])
log_recall <- log_cm[2,2] / sum(log_cm[2,])
log_f1_score <- 2 * log_precision * log_recall / (log_precision + log_recall)

# Print results
cat("Logistic Regression Model Results:\n")
cat("Confusion Matrix:\n")
print(log_cm)
cat(paste0("Accuracy: ", round(log_accuracy, 4), "\n"))
cat(paste0("Precision: ", round(log_precision, 4), "\n"))
cat(paste0("Recall: ", round(log_recall, 4), "\n"))
cat(paste0("F1 Score: ", round(log_f1_score, 4), "\n"))

#Decision tree
# Create & visualize decision tree model
model_DT <- rpart(HeartDisease ~., data=train_set, method="class")
rpart.plot(model_DT)
# Model Evaluation
# Compare model prediction outcome to result from test set
predict_DT <- predict(model_DT, test_set, type="class")

# Confusion Matrix
cm_DT <- table(test_set$HeartDisease, predict_DT)

# Calculate evaluation metrics
TN_DT <- cm_DT[1, 1]  # True Negative (correctly predict people with no heart disease (HD))
FP_DT <- cm_DT[1, 2]  # False Positive (incorrectly classify people with HD under no)
FN_DT <- cm_DT[2, 1]  # False Negative (incorrectly classify people with no HD as yes)
TP_DT <- cm_DT[2, 2]  # True Positive (correctly predict people with HD)

# Measure model performance 
accuracy_DT <- (TP_DT + TN_DT) / sum(cm_DT) # Accuracy
precision_DT <- TP_DT / (TP_DT + FP_DT) # Precision
recall_DT <- TP_DT / (TP_DT + FN_DT)  # Recall(Sensitivity)
f1_DT <- 2 * (precision_DT * recall_DT) / (precision_DT + recall_DT)  # F1 Score

# Print result
print(cm_DT)
print(paste('Accuracy: ', round(accuracy_DT, 4)))
print(paste('Precision: ', round(precision_DT, 4)))
print(paste('Recall: ', round(recall_DT, 4)))
print(paste('F1 Score: ', round(f1_DT, 4)))

#Gradient boosting classifier
#Copy train and test dataset as new dataset to avoid overwrite
train_set2 <- train_set
test_set2 <- test_set

# Convert the Species column to a factor variable
train_set2$HeartDisease <- as.factor(train_set2$HeartDisease)
test_set2$HeartDisease <- as.factor(test_set2$HeartDisease)

# Combine the factor levels from both datasets
levels(test_set2$HeartDisease) <- levels(train_set2$HeartDisease)

# fit GBM model on training set
gbm_model <- train(HeartDisease ~ ., data = train_set2, method = "gbm", verbose = FALSE)

# make predictions on testing set
gbm_predicted <- predict(gbm_model, newdata = test_set2)

# create confusion matrix and calculate performance metrics
gbm_confusion <- confusionMatrix(gbm_predicted, test_set2$HeartDisease)
gbm_accuracy <- gbm_confusion$overall['Accuracy']
gbm_precision <- gbm_confusion$byClass['Pos Pred Value']
gbm_recall <- gbm_confusion$byClass['Sensitivity']
gbm_f1_score <- gbm_confusion$byClass["F1"]

# Print result
gbm_confusion$table
print(paste('Accuracy: ', round(gbm_accuracy, 4))) 
print(paste('Precision: ', round(gbm_precision, 4)))
print(paste('Recall: ', round(gbm_recall, 4)))
print(paste('F1 Score: ', round(gbm_f1_score, 4)))

#KNN
#Copy train and test dataset as new dataset to avoid overwrite
train_set3 <- train_set
test_set3 <- test_set

# Combine the factor levels from both datasets
levels(test_set2$HeartDisease) <- levels(train_set2$HeartDisease)

train_set3$HeartDisease <- factor(train_set3$HeartDisease, levels = c(0, 1))
test_set3$HeartDisease <- factor(test_set3$HeartDisease, levels = c(0, 1))

# Specify the number of neighbors (k)
k <- 5

# Train the KNN model
knn_model <- knn(train = train_set3[, 1:4], test = test_set3[, 1:4], cl = train_set3$HeartDisease, k = k)

# generate confusion matrix
knn_conf_matrix <- confusionMatrix(knn_model, test_set3$HeartDisease)
knn_accuracy <- knn_conf_matrix$overall['Accuracy']
knn_recall <- knn_conf_matrix$byClass['Sensitivity']
knn_precision <- knn_conf_matrix$byClass['Pos Pred Value']
#knn_f1_score <- 2 * (precision * recall) / (precision + recall)
knn_f1_score <- knn_conf_matrix$byClass["F1"]

# Print result
print(knn_conf_matrix$table)
print(paste('Accuracy: ', round(knn_accuracy, 4))) 
print(paste('Precision: ', round(knn_precision, 4)))
print(paste('Recall: ', round(knn_recall, 4)))
print(paste('F1 Score: ', round(knn_f1_score, 4)))

#Support Vector MAchines(SVM)
# Convert the Species column to a factor variable
# Using Train set 2 and test set 2 which already converted.

# Train an SVM model with a linear kernel
svm_model <- svm(HeartDisease ~ ., data = train_set2, kernel = "linear")

# Make predictions on the test set
svm_predicted <- predict(svm_model, test_set2)

# Show the confusion matrix and performance metrics
svm_cm <- confusionMatrix(svm_predicted, test_set2$HeartDisease)
svm_accuracy <- svm_cm$overall["Accuracy"]
svm_precision <- svm_cm$byClass["Precision"]
svm_recall <- svm_cm$byClass["Recall"]
svm_f1_score <- svm_cm$byClass["F1"]

# Print result
print(svm_cm)
print(paste('Accuracy: ', round(svm_accuracy, 4))) 
print(paste('Precision: ', round(svm_precision, 4)))
print(paste('Recall: ', round(svm_recall, 4)))
print(paste('F1 Score: ', round(svm_f1_score, 4)))

#Random Forest
# Create random forest model
model_RF <- randomForest(HeartDisease ~ ., data = train_set, ntree = 100)
# Model Evaluation
# Compare model prediction outcome to the result from the test set
predict_RF <- predict(model_RF, test_set, type = "class")
compare_RF <- table(test_set$HeartDisease, predict_RF)
# compare_RF
# Output: The table will display the comparison between the actual values and the predicted values (confusion matrix)

# Measure model performance (Accuracy)
acc_test <- sum(diag(compare_RF)) / sum(compare_RF)
print(paste('Accuracy: ', round(acc_test, 4)))
# Print variable importance
var_importance <- importance(model_RF)
print(var_importance)
# Output: Importance measures for each variable in the random forest model

# Plot the random forest - IncNodePurity
varImpPlot(model_RF)
# Plot the random forest - line plot
plot(model_RF)

#XGBoost Model
# Ensure that the target variable (HeartDisease) is encoded as 0 and 1
train_set2$HeartDisease <- as.numeric(train_set2$HeartDisease) - 1
test_set2$HeartDisease <- as.numeric(test_set2$HeartDisease) - 1

# Create the data matrix and label vector
train_data <- data.matrix(train_set2[, -which(names(train_set2) == "HeartDisease")])
train_label <- train_set2$HeartDisease

test_data <- data.matrix(test_set2[, -which(names(test_set2) == "HeartDisease")])
test_label <- test_set2$HeartDisease

# Train the XGBoost Random Forest model
xgbrf_model <- xgboost(data = train_data,
                       label = train_label,
                       objective = "binary:logistic",
                       booster = "gbtree",
                       eval_metric = "logloss",
                       nrounds = 100)

# Make predictions on the test set
xgbrf_predicted <- predict(xgbrf_model, test_data)

# Convert predictions to binary values (0 or 1)
xgbrf_predicted <- as.numeric(xgbrf_predicted > 0.5)

# Calculate performance metrics
xgbrf_cm <- confusionMatrix(as.factor(xgbrf_predicted), as.factor(test_label))
xgbrf_accuracy <- xgbrf_cm$overall["Accuracy"]
xgbrf_precision <- xgbrf_cm$byClass["Precision"]
xgbrf_precision <- round(as.numeric(xgbrf_precision), 4)
xgbrf_recall <- xgbrf_cm$byClass["Recall"]
xgbrf_recall <- round(as.numeric(xgbrf_recall), 4)
xgbrf_f1_score <- xgbrf_cm$byClass["F1"]
xgbrf_f1_score <- round(as.numeric(xgbrf_f1_score), 4)

# Print results
print(xgbrf_cm)
print(paste("Accuracy:", round(xgbrf_accuracy, 4)))
print(paste("Precision:", xgbrf_precision))
print(paste("Recall:", xgbrf_recall))
print(paste("F1 Score:", xgbrf_f1_score))

#Model Evaluation
# Define the models and their corresponding evaluation metrics
models <- c("Logistic Regression", "Decision Tree", "GradientBoostingClassifier", "KNN","SVM", "XGBoost")
accuracy <- c(log_accuracy, accuracy_DT, gbm_accuracy, knn_accuracy, svm_accuracy,xgbrf_accuracy)
precision <- c(log_precision, precision_DT , gbm_precision, knn_precision, svm_precision,xgbrf_precision)
recall <- c(log_recall, recall_DT, gbm_recall, knn_recall, svm_recall,xgbrf_recall)
f1_score <- c(log_f1_score, f1_DT, gbm_f1_score, knn_f1_score, svm_f1_score,xgbrf_f1_score)

# Create a summary table
summary_table <- data.frame(Model = models, Accuracy = accuracy, Precision = precision,Recall = recall, F1_Score = f1_score)

# Print the summary table
kable(summary_table)
# Data (assuming you have the data in a data frame)
data <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Gradient Boosting Classifier", "KNN", "SVM", "XGBoost"),
  Accuracy = c(0.8530806, 0.8483412, 0.8436019, 0.7819905, 0.8388626, 0.8530806),
  Precision = c(0.8764045, 0.8173077, 0.8389831, 0.7815126, 0.8434783, 0.8661000),
  Recall = c(0.7959184, 0.8673469, 0.8761062, 0.8230088, 0.8584071, 0.8584000),
  F1_Score = c(0.8342246, 0.8415842, 0.8571429, 0.8017241, 0.85087721, 0.8622000)
)

# Melt the data for easier plotting
data_molten <- melt(data, id.vars = "Model")

# Convert variable to factor with ordered levels
data_molten$variable <- factor(data_molten$variable, levels = c("Accuracy", "Precision", "Recall", "F1_Score"))

# Create the line chart
ggplot(data_molten, aes(x = Model, y = value, color = variable, group = variable)) +
  geom_line() +
  labs(title = "Model Performance Comparison", x = "Model", y = "Metric Value") +
  theme_minimal()  

#Making a final metric
log_metric=(log_accuracy*log_f1_score)/log_precision
dt_metric=(accuracy_DT*f1_DT)/precision_DT
gbm_metric=(gbm_accuracy*gbm_f1_score)/gbm_precision
knn_metric=(knn_accuracy*knn_f1_score)/knn_precision
svm_metric=(svm_accuracy*svm_f1_score)/svm_precision
xgbrf_metric=(xgbrf_accuracy*xgbrf_f1_score)/xgbrf_precision

#average Metric
avg_metric=(log_metric+dt_metric+gbm_metric+knn_metric+svm_metric+xgbrf_metric)/6

log_ratio=log_metric/avg_metric
dt_ratio=dt_metric/avg_metric
gbm_ratio=gbm_metric/avg_metric
knn_ratio=knn_metric/avg_metric
svm_ratio=svm_metric/avg_metric
xgbrf_ratio=xgbrf_metric/avg_metric

