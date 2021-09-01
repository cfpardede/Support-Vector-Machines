#### Cesar Pardede ####
# Midterm (combined)
# svm_examples.R and Pardede_533_Midterm.R
# Math 533
# 2020-10-19


########################
#### svm_examples.R ####
########################


library(e1071)

x1 = rnorm(100, 7, 1)
x2 = rnorm(100, 2, 2)
y1 = rnorm(100, 7, 1)
y2 = rnorm(100, 2, 1)

data1 = data.frame(x = x1, y = y1, class = "1")
data2 = data.frame(x = x2, y = y2, class = "2")
data = rbind(data1, data2)

plot(data1$x, data1$y, col = 2, pch = 20, xlim = c(0, 10), ylim = c(0, 10), xlab = 'x', ylab = 'y', main = 'Randomly-generated Data')
points(data2$x, data2$y, col = 4, pch = 20)
legend('topleft', c('class 1', 'class 2'), pch = c(20, 20), col = c(2, 4))

linear_svm = svm(class ~ y + x, data = data, kernel = 'linear')
polynom_svm = svm(class ~ y + x, data = data, kernel = 'polynomial')
radial_svm = svm(class ~ y + x, data = data, kernel = 'radial')
sigmoid_svm = svm(class ~ y + x, data = data, kernel = 'sigmoid')

plot(linear_svm, data = data)
plot(polynom_svm, data = data)
plot(radial_svm, data = data)
plot(sigmoid_svm, data = data)

linear_svm1 = svm(class ~ y + x, data = data, kernel = 'linear', cost = 10)
linear_svm1 = svm(class ~ y + x, data = data, kernel = 'linear', cost = 1)
linear_svm3 = svm(class ~ y + x, data = data, kernel = 'linear', cost = 0.1)
linear_svm4 = svm(class ~ y + x, data = data, kernel = 'linear', cost = 0.01)

plot(linear_svm1, data = data)
plot(linear_svm2, data = data)
plot(linear_svm3, data = data)
plot(linear_svm4, data = data)

x3 = rnorm(100, 9, 1)
y3 = rnorm(100, 2, 0.5)
data3 = data.frame(x = x3, y = y3, class = "3")
data = rbind(data, data3)

plot(data1$x, data1$y, col = 2, pch = 20, xlim = c(0, 10), ylim = c(0, 10), xlab = 'x', ylab = 'y', main = 'Randomly-generated Data')
points(data2$x, data2$y, col = 4, pch = 20)
points(data3$x, data3$y, col = 1, pch = 20)
legend('topleft', c('class 1', 'class 2', 'class 3'), pch = c(20, 20, 20), col = c(2, 4, 1))

linear_svm5 = svm(class ~ y + x, data = data, kernel = 'linear', cost = 1)
polynom_svm2 = svm(class ~ y + x, data = data, kernel = 'polynomial')
radial_svm2 = svm(class ~ y + x, data = data, kernel = 'radial')
sigmoid_svm2 = svm(class ~ y + x, data = data, kernel = 'sigmoid')

plot(linear_svm5, data = data)
plot(polynom_svm2, data = data)
plot(radial_svm2, data = data)
plot(sigmoid_svm2, data = data)


###############################
#### Pardede_533_Midterm.R ####
###############################


# include e1071, contains svm function
library(e1071) # for SVM
library(gbm) # for boosting
library(randomForest) # for random forests

#### VARIABLE SELECTION AND DATA CLEANING ####
# read in data. we'll use 10000 samples for training, and randomly sample
# another 2000 samples for testing
n = 10000
train_data <- data.frame(read.csv('Fall 2020/Math 533/Midterm/adult.data')[1:n, ])
test_data <- data.frame(read.csv('Fall 2020/Math 533/Midterm/adult.data')[n:(n+2000),])
predictors = c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 
               'marital_status', 'occupation', 'relationship', 'race', 
               'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
               'native_country', 'income')
names(train_data) = predictors
names(test_data) = predictors

# we don't need fnlwgt or education
train_data <- train_data[, names(train_data) != 'education' & names(train_data) != 'fnlwgt']
test_data <- test_data[, names(test_data) != 'education' & names(test_data) != 'fnlwgt']
predictors = predictors[predictors != 'education' & predictors != 'fnlwgt']


#### EDA ####
# we can get quick summaries of each variable if we want
summary(train_data) # imbalanced race
str(train_data)

# check for missing values
for (names in predictors){
  print(length(which(is.na(train_data[names]) == F)))
} # no missing values

for (names in predictors){
  print(length(train_data[train_data[names] == '?']))
} # no missing values marked as '?'

# select response and predictors
# we choose race here, but leave the response as variable and write code
# for a general response so we can make explore other responses later
response = 'race'
predictors = predictors[predictors != response]

#### CLASS IMBALANCE HANDLING ####
# create training sets
# we'll draw from the remaining data to make test data later. there are major 
# issues with class imbalances, but lets try balancing it ourselves

# approach #1: nothing - just read in the data as is
train_data1 = train_data

# approach #2: downsample/upsample combo - take equal number of samples
# and resamples for each class
num_classes = dim(unique(train_data[response]))
n_k = n/num_classes

ob_wh = sample(which(train_data['race'] == ' White'), n_k, replace = F)
ob_bl = sample(which(train_data['race'] == ' Black'),  n_k, replace = T)
ob_as = sample(which(train_data['race'] == ' Asian-Pac-Islander'),  n_k, replace = T)
ob_am = sample(which(train_data['race'] == ' Amer-Indian-Eskimo'),  n_k, replace = T)
ob_ot = sample(which(train_data['race'] == ' Other'),  n_k, replace = T)

train_data2 = train_data[c(ob_wh, ob_bl, ob_as, ob_am, ob_ot), ]
### NOTE: After considerable testing, it has been discovered that this uniform
# sampling method does not work. Every time we have generated SVMs from the 
# raw training data, and the uniformly sampled training data, the SVM generated
# by the uniformly sampled training data has ALWAYS performed worse as measured
# by accurate predictions using each SVM model and the same test set. We will 
# proceed with the rest of the code using train_data1 - the raw training data.
# A sample of some of the testing follows below.

# some testing comparing the effect of training on the raw training data and 
# the combo training data. 
lin_svm = svm(formula, data = train_data1, kernel = 'linear', cross = 5, cost = 1); lin_svm1$accuracies
lin_svm2 = svm(formula, data = train_data2, kernel = 'linear', cross = 5, cost = 1); lin_svm2$accuracies
# combo sampling performed worse  with a linear kernel

lsvm1_pred = predict(lin_svm1, test_data)
table(lsvm1_pred, test_data[, 'race'])
lsvm2_pred = predict(lin_svm2, test_data)
table(lsvm2_pred, test_data[, 'race'])
# interesting results

poly_svm1 = svm(formula, data = train_data1, kernel = 'polynomial', cross = 5); poly_svm1$accuracies
poly_svm2 = svm(formula, data = train_data2, kernel = 'polynomial', cross = 5); poly_svm2$accuracies

# combo sampling performed worse  with a polynomial kernel
psvm1_pred = predict(poly_svm1, test_data)
# table(psvm1_pred, test_data[, 'race'])
psvm2_pred = predict(poly_svm2, test_data)
length(which(psvm2_pred == test_data[, response]))/2000
# table(psvm2_pred, test_data[, 'race'])
# interesting results again

### NOTE: based on these results we decide to stop training with the 
# combo/uniform training data. Instead, just use the raw training data.

# base acc. if we guess white every time
length(which(train_data[, 'race'] == ' White'))/n # 0.8556


#### SVM KERNEL SELECTION AND PARAMETER TUNING ####
# svm; go through the process of finding best parameters for each kernel
formula = as.formula(paste(response, ' ~ ', paste(predictors, collapse = ' + ')))

cost_acc = list()
i = 1
for (c in c(10, 1, 0.1, 0.01)){
  lin_svm = svm(formula, data = train_data1, kernel = 'linear', cross = 5, cost = c); lin_svm$accuracies
  lsvm_pred = predict(lin_svm, test_data)
  cost_acc[[i]] = list(cost = c, acc = length(which(lsvm_pred == test_data[, response]))/2000)
  i = i + 1
} # best accuracy when cost = 1; 0.889

deg_acc = list()
i = 1
for (d in c(2, 3, 4, 5)){
  for (c in c(10, 1, 0.1, 0.01)){
    poly_svm = svm(formula, data = train_data1, kernel = 'polynomial', cross = 5, degree = d, cost = c); poly_svm$accuracies
    psvm_pred = predict(poly_svm, test_data)
    deg_acc[[i]] = c(deg = d, cost = c, acc = length(which(psvm_pred == test_data[, response]))/2000)
    i = i + 1
  }
} # accuracy stays unchanged 0.866 for different degrees and different cost values
# stay with default degrees and cost

rad_acc = list()
i = 1
for (g in c(0, 0.001, 0.01, 0.1)){
  for (c in c(10, 1, 0.1, 0.01)){
    rad_svm = svm(formula, data = train_data1, kernel = 'radial', cross = 5, gamma = g, cost = c); rad_svm$accuracies
    rsvm_pred = predict(rad_svm, test_data)
    rad_acc[[i]] = c(gam = g, cost = c, acc = length(which(rsvm_pred == test_data[, response]))/2000)
    print(rad_acc[[i]])
    i = i + 1
  }
}

gam_acc = list()
i = 1
for (g in c(0, 0.001, 0.01, 0.1)){
  for (c in c(0, 0.1, 1, 10)){
    poly_svm = svm(formula, data = train_data1, kernel = 'polynomial', cross = 5, gamma = g, coef0 = c); poly_svm$accuracies
    psvm_pred = predict(poly_svm, test_data)
    sig_svm = svm(formula, data = train_data1, kernel = 'sigmoid', cross = 5, gamma = g, coef0 = c)
    ssvm_pred = predict(sig_svm, test_data)
    gam_acc[[i]] = c(gam = g, coef = c, poly_acc = length(which(psvm_pred == test_data[, response]))/2000,
                     sig_acc = length(which(ssvm_pred == test_data[, response]))/2000)
    print(paste(g, c, i))
    i = i + 1
  }
}
# all this tuning didn't do much, accuracy remained the same as with default parameters
# the best performing SVM was the SVM with a linear kernel with default 
# parameter cost = 1, and accuracy of 0.889

#### COMPARE SVMS TO RANDOM FORESTS AND DECISION TREES (BOOSTING) ####

p = length(predictors)
i = 1
rf_list = list()
for (ntree in c(100, 500, 800)){
  for (m in c(floor(sqrt(p)), floor(p/2), p)){
    rf_model = randomForest(formula, data = train_data1, ntree = ntree, mtry = m)
    rf_pred = predict(rf_model, test_data, type = 'response')
    acc_rf = length(which(rf_pred == test_data[, response]))/2000
    rf_list[[i]] = c(ntree = ntree, mtry = m, acc = acc_rf)
    print(i)
    i = i + 1
  }
}# best performance by rf: acc = 0.8865, ntree = 500, mtry = 3

i = 1
gb_list = list()
for (ntree in c(100, 500, 800)){
  gb_model = gbm(formula, data = train_data1, cv.folds = 5, n.trees = ntree)
  gb_pred = predict(gb_model, test_data, n.trees = 100, type = 'response')
  gb_pred = apply(gb_pred, 1, which.max)
  gb_pred[gb_pred == 1] = ' Amer-Indian-Eskimo'
  gb_pred[gb_pred == 2] = ' Asian-Pac-Islander'
  gb_pred[gb_pred == 3] = ' Black'
  gb_pred[gb_pred == 4] = ' Other'
  gb_pred[gb_pred == 5] = ' White'
  acc_gb = length(which(gb_pred == test_data[, response]))/2000
  gb_list[[i]] = c(n.trees = ntree, acc = acc_gb)
  print(i)
  i = i + 1
}# best performance by gb: acc = 0.8905, ntree = 800

# best SVM, RF, and Boosting models
svm = svm(formula, data = train_data1, kernel = 'linear', cross = 5)
rf_model = randomForest(formula, data = train_data1, ntree = 500, mtry = 3)
gb_model = gbm(formula, data = train_data1, cv.folds = 5, n.trees = 800)

svm_pred = predict(svm, test_data)
rf_pred = predict(rf_model, test_data, type = 'response')
gb_pred = predict(gb_model, test_data, n.trees = 100, type = 'response')
gb_pred = apply(gb_pred, 1, which.max)
gb_pred[gb_pred == 1] = ' Amer-Indian-Eskimo'
gb_pred[gb_pred == 2] = ' Asian-Pac-Islander'
gb_pred[gb_pred == 3] = ' Black'
gb_pred[gb_pred == 4] = ' Other'
gb_pred[gb_pred == 5] = ' White'

svm_acc = length(which(svm_pred == test_data[, response]))/2000
rf_acc = length(which(rf_pred == test_data[, response]))/2000 
gb_acc = length(which(gb_pred == test_data[, response]))/2000

accuracy = list(svm = svm_acc, rf = rf_acc, gb = gb_acc)
accuracy

table(svm_pred, test_data[, 'race'])
table(rf_pred, test_data[, 'race'])
table(gb_pred, test_data[, 'race'])
