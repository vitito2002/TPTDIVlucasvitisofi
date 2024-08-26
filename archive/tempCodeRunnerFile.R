
setwd("/Users/victorpablonavajas/Desktop/fq/TD6/TPTDIVlucasvitisofi/archive")
# Especifica un mirror de CRAN
options(repos = c(CRAN = "https://cloud.r-project.org"))
# Cargar librerías necesarias
install.packages("dplyr")
install.packages("Metrics")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("MLmetrics")
install.packages("pROC")
library(pROC)
library(MLmetrics)
library(rpart)
library(rpart.plot)
library(dplyr)


datos_originales <- read.csv("diabetes_binary_health_indicators_BRFSS2015.csv")
indices <- sample(1:nrow(datos_originales), 50000)
df<- datos_originales[indices,]

### EJERCICIO 2 ###
# Convertimos la variable 'GenHealth' en categorica
df$GenHlth_cat <- factor(df$GenHlth)
df$GenHlth_cat <- cut(df$GenHlth, breaks = c(0,1, 2, 3, 4, 5), labels = c("excellent", "very good", "good" , "fair", "poor"))

# Convertimos la variable 'Education' en categorica
df$Education_Category <- factor(df$Education)
df$Education_Category <- cut(df$Education,breaks = c(0, 1, 2, 3, 4, 5, 6), labels = c("No School", "Primary", "High School", "High School Graduate", "College", "College Graduate"))

# Convertimos la variable 'Income' en categorica
df$Income_Categorical <- factor(df$Income)
df$Income_Categorical <- cut(df$Income, breaks = c(0, 3, 6, 8), labels = c("Low", "Medium", "High"))


### EJERCICIO 3 ###
set.seed(42)

# creamos los índices 
n <- nrow(df)
indices <- sample(1:n, size = n)

train_size <- round(0.7 * n)
val_size <- round(0.15 * n)

# Particionamos los datos para que haya variables distintas en entrenamiento, training y validacion. (índices)
train_indices <- indices[1:train_size]
val_indices <- indices[(train_size + 1):(train_size + val_size)]
test_indices <- indices[(train_size + val_size + 1):n]

# Creamos los conjuntos de datos
train_data <- df[train_indices, ]
val_data <- df[val_indices, ]
test_data <- df[test_indices, ]

# entrenamos el modelo
arbol<-rpart( formula = Diabetes_binary ~. , data = train_data,  method = "class", control = rpart.control(cp = 0.001))

# predecimos con los datos de validación
validacion <- predict(arbol, newdata = val_data, type = "class")

# Calcular la accuracy en los datos de validación
accuracy_validation <- mean(validacion == val_data$Diabetes_binary)

# Imprimir la accuracy en los datos de validación
print(accuracy_validation)

# Predicciones en los datos de testeo
predicciones_test <- predict(arbol, newdata = test_data, type = "class")

# Calcular la accuracy en los datos de testeo
accuracy_test <- mean(predicciones_test == test_data$Diabetes_binary)

# Imprimir la accuracy en los datos de testeo
print(accuracy_test)
####################33
# testeo 
testeo <- predict(arbol, newdata = test_data, type="class")

# Comparar las predicciones con las etiquetas reales del conjunto de test
matriz_confusion_test <- table(test_data$Diabetes_binary, testeo)
print(matriz_confusion_test)

# Calcular métricas de rendimiento como accuracy (chat)
accuracy_test <- sum(diag(matriz_confusion_test)) / sum(matriz_confusion_test)

print(accuracy_test)
print(accuracy_validation)

# Graficar el árbol 
rpart.plot(arbol)

#############################Nuevos graficos by chat
# Instalar paquetes necesarios si no están instalados
#install.packages(c("ggplot2", "corrplot", "dplyr", "reshape2"))

# Cargar las librerías
library(ggplot2)
library(corrplot)
library(dplyr)
library(reshape2)

# Seleccionar las columnas numéricas del data frame
numeric_vars <- df %>% select_if(is.numeric)

# Calcular la matriz de correlación
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Graficar la matriz de correlación usando corrplot
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.5, addCoef.col = "black")

# Identificar las variables categóricas (factores o caracteres)
categorical_vars <- df %>% select_if(function(x) is.factor(x) | is.character(x))

# Convertir variables de tipo carácter a factores si es necesario
categorical_vars <- categorical_vars %>% mutate_if(is.character, as.factor)

# Si todavía no has instalado reshape2, lo necesitarás para usar melt:
# install.packages("reshape2")

# Convertir las variables categóricas a formato largo para ggplot2
melted_categorical <- melt(categorical_vars, id.vars = NULL)

# Graficar gráficos de barras para cada variable categórica
ggplot(melted_categorical, aes(x = value)) +
  geom_bar(fill = "skyblue") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Categoría", y = "Frecuencia", title = "Distribución de Variables Categóricas")


# Precisión
precision_test <- matriz_confusion_test[2, 2] / sum(matriz_confusion_test[, 2])

# Recall
recall_test <- matriz_confusion_test[2, 2] / sum(matriz_confusion_test[2, ])

# F1-score
f1_score_test <- 2 * (precision_test * recall_test) / (precision_test + recall_test)

# Calcular probabilidades para AUC-ROC
probabilidades_test <- predict(arbol, newdata = test_data, type = "prob")[, 2]

# AUC-ROC
roc_test <- roc(test_data$Diabetes_binary, probabilidades_test)
auc_roc_test <- auc(roc_test)

print(paste("Accuracy test matriz de confusion", accuracy_test))
#print(paste("Acurracy testeo", accuracy_testeo))
print(paste("Acurracy Validation", accuracy_validation))
print(paste("Precisión test", precision_test))
print(paste("Recall test", recall_test))
print(paste("F1-score test", f1_score_test))
print(paste("AUC-ROC test", auc_roc_test))
