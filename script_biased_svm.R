
datos = read.csv("wine_quality.csv", sep=";", header = TRUE)
datos$var_obj<-as.factor(as.numeric(datos$quality>=7))
library(plyr)
datos$var_obj=mapvalues(datos$var_obj,from=c("0","1"),to=c("malo","bueno"))
head(datos)

#modelo de svm radial con conjunto balanceado (no PU-Learning)
set.seed(1234)
ind <- sample(nrow(datos),floor(.75*nrow(datos)))
train <- datos[ind,]
test <- datos[-ind,]
train=train[,-12]

library(caret)
pos=train[train$var_obj=="bueno",]
neg=train[train$var_obj=="malo",]
set.seed(117349)
s=sample(1:nrow(neg),nrow(pos))
bal=rbind(pos,neg[s,])
grid=expand.grid(C=c(1:10),sigma=0.07950468)
svmNormal=train(var_obj~.,data=bal,method="svmRadial",trControl=trainControl("cv",10,classProbs=TRUE,
                                 summaryFunction=twoClassSummary),metric="ROC",tuneGrid=grid,seeds=117349)                                           

#crear datos de PU learning
set.seed(1234)
idx=sample(1:nrow(pos),500)
P=pos[idx,]
U=rbind(P[-idx,],neg)
U.var_obj=U$var_obj
U$var_obj=as.factor(rep("malo",nrow(U)))
library(e1071)
#generar 10-folds para validacion cruzada

#balancear U
set.seed(9999)
U.bal=U[sample(1:nrow(U),nrow(P)),]
PU.train=rbind(P,U.bal)
n=nrow(PU.train)

#validacion cruzada
set.seed(1234)
folds=floor(10*runif(n))+1
PU.train$folds=folds
vals=c(1)
metricas=vector()
ks=vector()
js=vector()
for(k in 1:1){
  for(j in 1:13){
    metrica=0
    for(i in 1:10){
      svm <- svm(var_obj~fixed_acidity+volatile_acidity+citric_acid+residual_sugar+chlorides+free_sulfur_dioxide+total_sulfur_dioxide+density+pH+sulphates+alcohol,
                     data=PU.train[PU.train$folds!=i,], type='C-classification', kernel='radial', class.weights=data.frame(malo=vals[k],bueno=2^j))
      #calificar conj. de validacion
      labels=predict(svm,PU.train[PU.train$fold==i,])
      #matriz de confusion
      confMat=table(labels,PU.train[PU.train$folds==i,"var_obj"])
      #calcular r^2/p(f(x)=1)
      precision=confMat[2,2]/(confMat[2,2]+confMat[2,1])
      recall=confMat[2,2]/(confMat[2,2]+confMat[1,2])
      #r=confMat[2,2]^2*confMat[1,2]*confMat[2,1]/(confMat[2,2]*confMat[1,2]+confMat[2,2]*confMat[2,1])
      #metrica=metrica+r^2/(confMat[2,2]+confMat[2,1]/(confMat[2,2]+confMat[2,1]+confMat[1,1]+confMat[1,2]))
      metrica=2*precision*recall/(precision+recall)
    }
    ks=c(ks,k)
    js=c(js,j)
    metricas=c(metricas,metrica)
  }
}
CV=as.data.frame(cbind(ks,js,metricas))
names(CV)=c("ks","js","metricas")
CV$ks=vals[CV$ks]
CV$js=2^CV$js

#c_0=1, c_1=2
#entrenar el mejor modelo
PU.svm <- svm(var_obj~fixed_acidity+volatile_acidity+citric_acid+residual_sugar+chlorides+free_sulfur_dioxide+total_sulfur_dioxide+density+pH+sulphates+alcohol,
           data=PU.train, type='C-classification', kernel='radial', class.weights=data.frame(malo=1,bueno=2),probability=TRUE)

#comparar AUC de ambos modelos 
library(ROCR)
pred=prediction(attr(predict(PU.svm,test,probability=TRUE),"prob")[,2],test$var_obj)
perf=performance(pred,"tpr","fpr")
roc.pu.x=unlist(attr(perf,"x.values"))
roc.pu.y=unlist(attr(perf,"y.values"))

pred=prediction(predict(svmNormal,test,type="prob")[,1],test$var_obj)
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
  theme(legend.position=c(0.7,0.5),legend.key.size=3)

library(gridExtra)
grid.arrange(graficaroc,graficalift,nrow=1,ncol=2)


#sacar metricas de la matriz de confusion
pu.mc=table(predict(PU.svm,test),test$var_obj)
normal.mc=table(predict(svmNormal,test),test$var_obj)

pu.mc
normal.mc

pdf("PU_ROC.pdf",width=4,height=4)
graficaroc
dev.off()

pdf("PU_LIFT.pdf",width=4,height=4)
graficalift
dev.off()

library(xtable)
