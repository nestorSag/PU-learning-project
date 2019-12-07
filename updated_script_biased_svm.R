packrat::restore()

library(dplyr)
library(caret)
library(e1071)
library(kernlab)

#cargar datos y crear variable objetivo
datos = read.csv("wine_quality.csv", sep=";", header = TRUE) %>%
    mutate(var_obj = ifelse(quality >=7,"bueno","malo"))


#modelo de svm radial con conjunto balancedanceado (no PU-Learning)
set.seed(1234)
train_proportion = 0.75
train_idx = sample(nrow(datos),floor(train_proportion*nrow(datos)))
train = datos[train_idx,]
test = datos[-train_idx,]

#quitar outlier
train=train[,-12]

pos = train %>% filter(var_obj == "bueno")
neg = train %>% filter(var_obj == "malo")

#crear conjunto de datos balancedanceado 50% positivo -50% negativo
# svm no acepta pesos para las observaciones, asi que un submuestreo es necesario
s=sample(1:nrow(neg),nrow(pos))
balanced=rbind(pos,neg[s,])
grid=expand.grid(C=c(1:10),sigma=0.07950468) # creo que este valor de sigma lo obutve con la funcion kernlab::sigest 
svmNormal=train(var_obj~.,data=balanced,method="svmRadial",trControl=trainControl("cv",10,classProbs=TRUE,
                                 summaryFunction=twoClassSummary),metric="ROC",tuneGrid=grid,seeds=117349)                                           

#crear datos de PU learning
n_pu = 500
pu_idx=sample(1:nrow(pos),n_pu)
P=pos[pu_idx,]
U=rbind(P[-pu_idx,],neg) %>% mutate(var_obj = "desconocido")
#generar 10-folds para validacion cruzada

#balancear U
u_subsample=U[sample(1:nrow(U),nrow(P)),]
pu_train=rbind(P,u_subsample)
n=nrow(pu_train)

#validacion cruzada
folds=floor(10*runif(n))+1
pu_train = pu_train %>% mutate(folds = floor(10*runif(n))+1)
vals=c(1)
metricas=vector()
ks=vector()
js=vector()
model_formula = var_obj~fixed_acidity+volatile_acidity+citric_acid+residual_sugar+chlorides+free_sulfur_dioxide+total_sulfur_dioxide+density+pH+sulphates+alcohol
for(k in 1:1){
  for(j in 1:13){
    metrica=0
    for(fold_idx in 1:10){

      train_folds = pu_train %>% filter(folds != fold_idx)
      test_fold = pu_train %>% filter(folds == fold_idx)

      svm <- svm(
        model_formula,
        data=train_folds, 
        type='C-classification', 
        kernel='radial', 
        class.weights=data.frame(desconocido=vals[k],bueno=2^j))

      #calificar conj. de validacion
      labels=predict(svm,test_fold)
      #matriz de confusion
      confMat=table(labels,test_fold[["var_obj"]])
      #calcular r^2/p(f(x)=1)
      precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
      recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])

      metrica=2*precision*recall/(precision+recall) #F1 score (https://en.wikipedia.org/wiki/F1_score)
    }
    ks=c(ks,k)
    js=c(js,j)
    metricas=c(metricas,metrica)
  }
}

# crear data frame con metricas de validacion cruzada
CV=as.data.frame(cbind(ks,js,metricas))
names(CV)=c("ks","js","metricas")
CV = CV %>% mutate(ks = vals[ks], 2^js)

# encontrar los mejores hiperparametros
row_max = which.max(CV[["metricas"]])
best_k = CV[row_max,"ks"]
best_j = CV[row_max,"js"]
#c_0=1, c_1=2
#entrenar el mejor modelo
PU.svm <- svm(
           formula=model_formula,
           data=pu_train, 
           type='C-classification', 
           kernel='radial', 
           class.weights=data.frame(desconocido=best_k,bueno=2^best_j),
           probability=TRUE)

#comparar AUC de ambos modelos 
library(ROCR)
pred=prediction(attr(predict(PU.svm,test,probability=TRUE),"prob")[,2],test$var_obj)
perf=performance(pred,"tpr","fpr")
roc.pu.x=unlist(attr(perf,"x.values"))
roc.pu.y=unlist(attr(perf,"y.values"))

pred=prediction(predict(svmNormal,test,type="prob")[,2],test$var_obj)
perf=performance(pred,"tpr","fpr")
roc.normal.x=unlist(attr(perf,"x.values"))
roc.normal.y=unlist(attr(perf,"y.values"))

rocsvm=data.frame(ejeX=c(roc.pu.x,roc.normal.x),ejeY=c(roc.pu.y,roc.normal.y),
               Metodo=c(rep("Biased SVM",length(roc.pu.x)),rep("SVM estándar",length(roc.normal.x))))

graficaroc=ggplot(data=rocsvm,aes(x=ejeX,y=ejeY))+geom_line(aes(colour=Metodo),size=1.5)+xlab("1-especificidad")+
  ylab("sensibilidad")+theme_classic()+theme(legend.position=c(0.5,0.5))

#comparar lift de ambos modelos 
pred=prediction(attr(predict(PU.svm,test,probability=TRUE),"prob")[,2],test$var_obj)
perf=performance(pred,"lift","rpp")
lift.pu.x=unlist(attr(perf,"x.values"))
lift.pu.y=unlist(attr(perf,"y.values"))

pred=prediction(predict(svmNormal,test,type="prob")[,1],test$var_obj)
perf=performance(pred,"lift","rpp")
lift.normal.x=unlist(attr(perf,"x.values"))
lift.normal.y=unlist(attr(perf,"y.values"))

liftsvm=data.frame(ejeX=c(lift.pu.x,lift.normal.x),ejeY=c(lift.pu.y,lift.normal.y),
                  Metodo=c(rep("Biased SVM",length(lift.pu.x)),rep("SVM estándar",length(lift.normal.x))))

graficalift=ggplot(data=liftsvm,aes(x=ejeX,y=ejeY))+geom_line(aes(colour=Metodo),size=1.5)+xlab("P(y=1|x)")+
  ylab("Lift")+ylim(1,1.3)+xlim(1,0.01)+theme_classic()+ 
  theme(legend.position=c(0.7,0.5))

library(gridExtra)
grid.arrange(graficaroc,graficalift,nrow=1,ncol=2)


#sacar metricas de la matriz de confusion
pu.mc=table(predict(PU.svm,test),test$var_obj)
normal.mc=table(predict(svmNormal,test),test$var_obj)

pu.mc
normal.mc

## guardar imagenes en PDF
pdf("PU_ROC.pdf",width=4,height=4)
graficaroc
dev.off()

pdf("PU_LIFT.pdf",width=4,height=4)
graficalift
dev.off()
