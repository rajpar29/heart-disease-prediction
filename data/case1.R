hd <- read.table("hd.txt", header = FALSE, sep = ',')
set.seed(1234)
#creating a secondary df that is numerical
hd.num <- read.table("hd.txt", header = FALSE, sep = ',')
summary(hd)
# V14 changed to "No" for any "0" item and "Yes" for any positive heart disease (1,2,3,4)
hd$V14 <- as.factor(ifelse(hd$V14 >=1, "Yes", "No"))
#creating an index upon which to split data into train and test sets
ind <- sample(2, nrow(hd), replace=TRUE, prob=c(0.8, 0.2))
train <- hd[ind==1,]
test <- hd[ind==2,]

library(rpart)
library(randomForest)
library(caret)
#singular model tree 
model.tree <- rpart(V14 ~ . , data=train)
pred.model.tree <- predict(model.tree, test, type = "class")
table(pred.model.tree,test$V14)
t <- table(pred.model.tree,test$V14)
#examing the missclassifcation cases
confusionMatrix(t)

plot(model.tree)
text(model.tree)
#creating a df sample with replacement
samplehd <- sample(nrow(train), replace = T)
u <- unique(samplehd)
#confirming the sampling process was completed
length(u)
length(samplehd)
#testing different forest depths, the ideal # was 75
randF <- randomForest(V14 ~., data = train, ntree = 75)
pred.randF <- predict(randF, test, type = "class")
tf <- table(pred.randF,test$V14)
confusionMatrix(tf)
#univariate analysis, looking for outliers
boxplot(hd.num$V14)
hist(hd.num$V14)
#density plot to see what the majority of V14 was noted to be
d <- density(hd.num$V14)
plot(d, col="red")
#density plot of age
d1 <- density(hd.num$V1)
plot(d1, col="red")
hd.num$V12 <-as.numeric(hd.num$V12)
hd.num$V13 <-as.numeric(hd.num$V13)
#PCA on variables to see which account for greatest variance
hd.pca<- prcomp(hd.num, center = TRUE, scale. = TRUE)
print(hd.pca)
plot(hd.pca)
#Establishing a cut-off point for PCs
abline(h=1, col="red")
#Examing correlation via the Pearson method with obmission of "NA" values
cor(hd.num, use = "complete.obs", method = "pearson")
library(corrgram)
#Visualization of correlation between variables
corrgram(hd.num, order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="Correlation of Variables r/t Heart Disease")
plot(hd.num$V14~., data = hd.num)
#Linear model with all variables used to predict V14
fit <- lm(V14 ~., data = hd.num)
summary(fit)
#Identifying the best combination of variables that explain the greatest amount of variance
stephd <- regsubsets(hd.num$V14~., data = hd.num, nbest = 10)
summary(stephd)
#Plotting off of adjusted R^2 
plot(stephd, scale = "adjr2")
#Examing relative importance of each variable on the model as well as a secondary variance accountability check
library(leaps)
library(relaimpo)
calc.relimp(fit,type=c("lmg","last","first","pratt"),
            rela=TRUE)
