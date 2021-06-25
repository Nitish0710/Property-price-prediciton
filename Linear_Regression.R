#package loading
library(car)
library(DescTools)
library(corrplot)

#check working directory
getwd()

#setting working directory
setwd("E:/Data science/linear regression in R/")

#read data
TrainRaw = read.csv("PropertyPrice_Train.csv",stringsAsFactors = TRUE)
TestRaw = read.csv("PropertyPrice_Prediction.csv",stringsAsFactors = TRUE)

View(TrainRaw)
View(TestRaw)

# Create Source Column in both Train and Test
TrainRaw$Source = "Train"
TestRaw$Source = "Test"

#combine data
FullRaw = rbind(TrainRaw,TestRaw)

#drop id column
FullRaw = subset(FullRaw,select = -Id)

View(FullRaw)

#check missing value
colSums(is.na(FullRaw))

#Garage
#Find missing values
#step 1 find mode [find mode for categorical and median for continoues variable]
Tempmode = Mode(TrainRaw$Garage,na.rm = TRUE)[1]
Tempmode

#step 2 find missing value
missingvalue = is.na(FullRaw[,"Garage"])

#step 3 impute values
FullRaw[missingvalue,"Garage"] = Tempmode
colSums(is.na(FullRaw))

#Garage_Built_Year
#Find missing values
#step 1 find median 
Tempmedian = median(TrainRaw$Garage_Built_Year,na.rm = TRUE)
Tempmedian

#step 2 find missing value
missingval = is.na(FullRaw[,"Garage_Built_Year"])

#step 3 impute value
FullRaw[missingval,"Garage_Built_Year"] = Tempmedian
colSums(is.na(FullRaw))

#EDA[correlation plot]
library(corrplot)
continous_variable_check = function(x)
{
  return(is.numeric(x)|is.integer(x))
}
continous_vars = sapply(TrainRaw,continous_variable_check)
continous_vars
#continous_vars contain all the variable which are continous in the data

corrDF = cor(TrainRaw[TrainRaw$Source == "Train",continous_vars])
View(corrDF)
windows()
corrplot(corrDF)

#dummy variable creation
Factorvars = sapply(FullRaw,is.factor)
Factorvars
dummyDF = model.matrix(~.,data = FullRaw[,Factorvars])
View(dummyDF)

dim(dummyDF)
FullRaw2 = cbind(FullRaw[,!Factorvars],dummyDF[,-1])

# Check the dimensions of FullRaw2
dim(FullRaw2)

# Check if all variables are now numeric/integer
str(FullRaw2) 

#sampling

# Step 1: Divide Fullraw2 into Train and Test
Train = subset(FullRaw2, subset = FullRaw2$Source == "Train", select = -Source)
PredictionDf = subset(FullRaw2, subset = FullRaw2$Source == "Test", select = -Source)


# Step 2: Divide Train further into Train and Test by random sampling
set.seed(123) # This is used to reproduce the SAME composition of the sample EVERYTIME
RowNumbers = sample(x = 1:nrow(Train), size = 0.80*nrow(Train))
head(RowNumbers)
Test = Train[-RowNumbers, ] # Testset
Train = Train[RowNumbers, ] # Trainset

dim(Train)
dim(Test)

#multicollinarity check
#remove variable with VIF<5

M1 = lm(Sale_Price ~ ., data = Train)

library(car)
sort(vif(M1), decreasing = TRUE)[1:3]

M2 = lm(Sale_Price ~ . -GarageAttchd,data = Train)
sort(vif(M2), decreasing = TRUE)[1:3]

M3 = lm(Sale_Price ~ . -GarageAttchd - Kitchen_QualityTA ,data = Train)
sort(vif(M3), decreasing = TRUE)[1:3]

M4 = lm(Sale_Price ~ . -GarageAttchd - Kitchen_QualityTA -First_Floor_Area,data = Train)
sort(vif(M4), decreasing = TRUE)[1:3]

# All variables are within the bound of VIF (VIF < 5). Now we can proceed towards model building and variable selection
summary(M3)

# Model optimization (by selecting ONLY significant variables through step() function)
############################

# Use step() function to remove insignificant variables from the model iteratively
M4 = step(M3) # Step function works on the concept of reducing AIC. Lower the AIC, better the model

summary(M4)

# Model diagnostics

# Few checks
# Homoskedasticity check
plot(M4$fitted.values, M4$residuals) 
# Should not show prominent non-constant variance (heteroskadastic) of errors against fitted values

# Normality of errors check
summary(M4$residuals) # To check the range. Will be used in histogram is next step
hist(M4$residuals, breaks = seq(-490000, 340000, 10000)) # Should be somewhat close to normal distribution

# Model Evaluation

# After doing all of this, the model has to be checked against the test data as well
# Lets predict on testset and then calculate a metric called MAPE to estimate the errors on testing data
M4_Pred = predict(M4, Test)
head(M4_Pred)
head(Test$Sale_Price)

Actual = Test$Sale_Price
Prediction = M4_Pred

# MAPE (Mean Absolute Percentage Error)
mean(abs((Actual - Prediction)/Actual))*100 # 16%
# This means on an "average", the house price prediction would have +/- error of 16%

# Generally, a MAPE under 10% is considered very good, and anything under 20% is reasonable.
# MAPE over 20% is usually not considered great.