load("~/UAB/4t/TFG/DadesEstudiAfil/Taules_dataframe.Rdata")

library(lubridate)
library(tidyverse)
library(dplyr)


afiliacio$DALTA <- as.Date(afiliacio$DALTA, format = "%d/%m/%Y")
afiliacio$DBAIXA <- as.Date(afiliacio$DBAIXA, format = "%d/%m/%Y")
registres <- rbind(select(baixa, 'ID_PERSONA', 'DALTA', 'DBAIXA'), select(afiliacio, 'ID_PERSONA', 'DALTA', 'DBAIXA'))
registres$DALTA <- as.Date(registres$DALTA, format = "%Y/%m/%d")
registres$DBAIXA <- as.Date(registres$DBAIXA, format = "%d/%m/%Y")



registres <- registres %>% arrange(DALTA, desc(DBAIXA))
registres <- registres %>% group_by(DALTA, ID_PERSONA) %>% slice(1) #eliminem casos repetits


# Crear función para concatenar ID_PERSONA con número de aparición
create_id <- function(x) {
  paste0(x, ".", seq_along(x))
}

registres$ID <- ave(registres$ID_PERSONA, registres$ID_PERSONA, FUN = create_id)
afiliacio$ID <- ave(afiliacio$ID_PERSONA, afiliacio$ID_PERSONA, FUN = create_id)

#NUMERO DE MES
registres$M_ALTA <- month(registres$DALTA)
registres$M_BAIXA <- month(registres$DBAIXA)

elapsed_months<-
function(end_date, start_date) {
  if(is.na(end_date)){
    ed <- as.POSIXlt(Sys.Date())
    sd <- as.POSIXlt(start_date)
    12 * (ed$year - sd$year) + (ed$mon - sd$mon)
  }
  else{
    ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
  }
}
# MESOS AFILIAT
registres$N_MONTH <- mapply(elapsed_months, registres$DBAIXA, registres$DALTA)

# NUMERO DE FORMACIONS  
#formacio$INICI <- as.Date(formacio$INICI, format = "%Y/%m/%d")
# prova <- registres %>% 
#   mutate(N_FORMACIO = sapply(ID_PERSONA, function(x) {
#     sum(x == formacio$ID_PERSONA & formacio$INICI >= DALTA & (is.na(DBAIXA) | formacio$INICI <= DBAIXA))
#   }))



# AFILIAT ACTUAL
AFILIAT<- ifelse(registres$ID_PERSONA %in% afiliats_actual$ID_PERSONA, 1, 0)
registres$AFILIAT<- AFILIAT

# Ens hem de quedar només amb els últims registres de cada persona? 
# 
### HI HA GENT QUE ES DONA D'ALTA I DE BAIXA EL MATEIX MES ??????


# N_AFILIACIONS

taula<-table(registres$ID_PERSONA)%>% data.frame()
names(taula)<- c('ID_PERSONA', 'N_AFILIACIONS')

registres <- merge(registres, taula, by= "ID_PERSONA")

# NUMERO DE FORMACIONS 
formacio$INICI <- as.Date(formacio$INICI, format = "%d/%m/%Y")
registres<-select(registres, -N_FORMACIONS)
registres <- registres %>%
  left_join(formacio, by = "ID_PERSONA") %>%
  filter(!is.na(INICI) & INICI >= DALTA & (is.na(DBAIXA) | INICI <= DBAIXA)) %>%
  group_by(ID_PERSONA, DALTA, DBAIXA) %>%
  summarize(N_FORMACIONS = n(), .groups = "drop") %>%
  right_join(registres, by = c("ID_PERSONA", "DALTA", "DBAIXA")) %>%
  select(ID_PERSONA, DALTA, DBAIXA, ID, M_ALTA, M_BAIXA, N_MONTH, N_FORMACIONS)

# NUMERO D'ASSESSORAMENTS

assessorament$DALTA <- as.Date(assessorament$DALTA, format = "%Y/%m/%d")
assessorament$INICI <- as.Date(assessorament$DALTA, format = "%Y/%m/%d")
assessorament <- dplyr::select(assessorament, -DALTA)

registres <- registres %>%
  left_join(assessorament, by = "ID_PERSONA") %>%
  filter(!is.na(INICI) & INICI >= DALTA & (is.na(DBAIXA) | INICI <= DBAIXA)) %>%
  group_by(ID_PERSONA, DALTA, DBAIXA) %>%
  summarize(N_ASSESSORAMENTS = n(), .groups = "drop") %>%
  right_join(registres, by = c("ID_PERSONA", "DALTA", "DBAIXA")) %>%
  select(ID_PERSONA, DALTA, DBAIXA, ID, M_ALTA, M_BAIXA, N_MONTH, N_FORMACIONS, N_ASSESSORAMENTS)




# ELECCIONS AL CENTRE
CENTRE_ELECCIONS<- ifelse(registres$ID_PERSONA %in% persones_amb_eleccions, 1, 0)
registres$CENTRE_ELECCIONS<- CENTRE_ELECCIONS


# SITUACIO LABORAL
registres <- merge(registres, select(afiliacio, c("ID_PERSONA", "SLABORAL", "ORIGEN")), by= "ID_PERSONA")

#CAUSA ULTIMA BAIXA (eliminar aquelles persones que la causa va ser mort)
causa_mort <- which(causa$DESCRIPCIO == "DEFUNCIÓ")
causa_mort <- causa[causa_mort, 1]
casos_eliminar<- unique(baixa[which(baixa$CAUSA == causa_mort), c("ID_PERSONA")])

baixa <- merge(baixa, causa, by= "CAUSA")

registres <- merge(registres, baixa[,c("ID_PERSONA","DBAIXA", "CAUSA")], by= c("ID_PERSONA", "DBAIXA"))


#EL CENTRE ON TREBALLA PERMET PAGAMENT PER NÒMINA (si,no)

centres_admeten_nomina <- empreses[which(empreses$NOMINA == "S"), c("CODICTREB")]
persones_centre_nomina <- subset(baixa, CODICTREB %in% centres_admeten_nomina)
persones_centre_nomina <- persones_centre_nomina$ID_PERSONA

CENTRE_NOMINA<- ifelse(registres$ID_PERSONA %in% persones_centre_nomina, 1, 0)
registres$CENTRE_NOMINA<- CENTRE_NOMINA


# SECTOR EMPRES I NÚMERO DE TREBALLADORS

persona_sector<- merge(select(empreses,c("CODICTREB", "SECTOR", "NTREBALL")),
                       select(baixa,c("ID_PERSONA", "CODICTREB")),
                       by= "CODICTREB")

registres <- merge(registres, 
                   select(persona_sector,c("ID_PERSONA", "SECTOR", "NTREBALL")), 
                   by= "ID_PERSONA")


# PERSONA HA ESTAT CANDIDATA
persones_candidates <- select(candidat, c("ID_PERSONA"))

CANDIDAT<- ifelse(registres$ID_PERSONA %in% persones_candidates, 1, 0)
registres$CANDIDAT<- CANDIDAT

# AFILIATS ACTUAL
AFILIAT_ACTUAL<- ifelse(registres$ID_PERSONA %in% afiliats_actual, 1, 0)
registres$AFILIAT_ACTUAL<- AFILIAT_ACTUAL


# PAGAMENTS RETORNATS DURANT L'AFILIACIÓ






write.csv(registres, "registres.csv", row.names = FALSE)




setwd("~/UAB/4t/TFG/DadesEstudiAfil")
registres<-read.csv("registres.csv")







# ELIMINO CASOS ON DBAIXA < DALTA I AQUELLS AMB DBAIXA > DACTUAL

registres <- registres %>% filter(registres$DBAIXA < Sys.Date()) 
registres <- registres %>% filter(registres$DBAIXA > registres$DALTA)



# Paso 1: Calcular el número de trimestres entre DALTA y DBAIXA
registres$DALTA <- as.Date(registres$DALTA, format = "%Y-%m-%d")
registres$DBAIXA <- as.Date(registres$DBAIXA, format = "%Y-%m-%d")
registres<-registres[order(registres$ID_PERSONA),]

registres$num_trimestres <- as.integer((registres$DBAIXA - registres$DALTA) / 91)  # Suposant que cada trimestre te 91 dies

# registres_repetidos<-data.frame()
# for (i in 1:nrow(registres)){
#   fila<-registres[i,]
#   df=data.frame()
#   for(rep in 1:fila$num_trimestres){df<-rbind(df,fila)}
#   registres_repetidos<-rbind(registres_repetidos, df)
#   print(i/284445*100)
# }
# 
# 
# 
registres <- registres %>% filter(registres$num_trimestres > 0) 
registres$ACTIVITAT<- ifelse(is.na(registres$N_ASSESSORAMENTS) & is.na(registres$N_FORMACIONS), "No", "Yes")

library(ggplot2)


ggplot(registres, aes(x = num_trimestres, fill = ACTIVITAT)) + 
  geom_histogram()

registres$num_trimestres<- as.factor(registres$num_trimestres)
dat<-table(registres$num_trimestres, registres$ACTIVITAT)

spineplot(dat, off = 0)



# 
# #VERSION 2
# registres_repetidos <- list()
# 
# for (i in 1:nrow(registres)) {
#   fila <- registres[i, ]
#   df <- data.frame()
#   
#   for (rep in 1:fila$num_trimestres) {
#     df <- rbind(df, fila)
#   }
#   
#   registres_repetidos[[i]] <- df
#   print(i / nrow(registres) * 100)
# }
# 
# registres_repetidos <- do.call(rbind, registres_repetidos)
# 
# 
# #VERSION 3:
# 
# registres_repetidos <- list()
# 
# for (i in 1:nrow(registres)) {
#   fila <- registres[i, ]
#   df <- data.frame(matrix(rep(fila, fila$num_trimestres), nrow = fila$num_trimestres, byrow = TRUE))
#   registres_repetidos[[i]] <- df
#   print(i / nrow(registres) * 100)
# }
# 
# registres_repetidos <- do.call(rbind, registres_repetidos)
# 
# 
# 
# 
# registres_repetidos$DALTA <- as.Date(registres_repetidos$DALTA, format = "%Y-%m-%d")
# registres_repetidos$fecha_fin_trimestre <- registres_repetidos$DALTA + (1:registres_repetidos$num_trimestres) * 91  # Suponiendo que cada trimestre tiene 91 días
# 


#Fem una proba amb menys registres:

registres<- registres[1:100,]

# VERSION 4

registres_repetidos <- lapply(1:nrow(registres), function(i) {
  fila <- registres[i, ]
  df <- data.frame(matrix(rep(fila, fila$num_trimestres), nrow = fila$num_trimestres, byrow = TRUE))
  print(i / nrow(registres) * 100)
  return(df)
})

library(data.table)

registres_new <- rbindlist(registres_repetidos)




library(lubridate)
library(tidyverse)
library(dplyr)

FIN_TRIMESTRE <- c()
# for (i in 1:nrow(df)) {
#   if (is.na(df$DBAIXA[[i]])) {
#     trimestres <- round(as.numeric(difftime(Sys.Date(), as.Date(df$DALTA[[i]]), units = "days")) / 30.44)
#   } else {
#     trimestres <- round(as.numeric(difftime(as.Date(df$DBAIXA[[i]]), as.Date(df$DALTA[[i]]), units = "days")) / 30.44) 
#   }
# 
#   if (trimestres > 0) {
#     fecha_inicio <- as.Date(df$DALTA[[i]]) + (0:(trimestres-1))*3*30
#     fecha_fin <- fecha_inicio + 3*30
#     FIN_TRIMESTRE[i] <- c(FIN_TRIMESTRE, fecha_fin)
#     
#   }
# }
names(registres_new)<- names(registres)

# for (i in 1:nrow(registres)) {
#   alta <- as.Date(registres$DALTA[[i]])
#   baixa <- as.Date(registres$DBAIXA[[i]])
#   fin_trimestre <- seq.Date(alta, baixa, by = "3 month")
#   fin_trimestre<-fin_trimestre[-1]
#   FIN_TRIMESTRE <- c(FIN_TRIMESTRE,fin_trimestre)
# }

# VERSIO 2
altas <- as.Date(registres$DALTA)
bajas <- as.Date(registres$DBAIXA)

fin_trimestre <- lapply(seq_along(altas), function(i) {
  seq.Date(altas[i], bajas[i], by = "3 months")[-1]
})
fin_trimestre<-c(fin_trimestre, rep(NA, 39))

FIN_TRIMESTRE <- do.call(c, fin_trimestre)

registres_new$FIN_TRIMESTRE<-FIN_TRIMESTRE

# BAIXA SEGÜENT TRIM:
BAIXA_TRIM<-c()
for (i in unique(registres$ID)){
  n = nrow(filter(registres_new, ID==i))
  BAIXA_TRIM<-c(BAIXA_TRIM, rep(0,n-1),1)
}

registres_new$BAIXA_TRIM <- BAIXA_TRIM


registres_new$ID_PERSONA <- do.call(c, registres_new$ID_PERSONA)
registres_new$DALTA <- do.call(c, registres_new$DALTA)
registres_new$N_FORMACIONS <- do.call(c, registres_new$N_FORMACIONS)
registres_new$N_ASSESSORAMENTS <- do.call(c, registres_new$N_ASSESSORAMENTS)

# NUMERO DE FORMACIONS  
registres_new_2 <- registres_new %>%
  left_join(formacio, by = "ID_PERSONA") %>%
  filter(!is.na(INICI) & INICI >= DALTA & (is.na(DBAIXA) |INICI <= FIN_TRIMESTRE)) %>%
  group_by(ID_PERSONA, DALTA, FIN_TRIMESTRE) %>%
  summarize(N_FORMACIONS_2 = n(), .groups = "drop") %>%
  right_join(registres_new, by = c("ID_PERSONA", "DALTA", "FIN_TRIMESTRE")) %>%
  select(ID_PERSONA, DALTA, FIN_TRIMESTRE, ID, M_ALTA, M_BAIXA, N_MONTH, N_FORMACIONS_2)
registres_new_2<-registres_new_2[order(registres_new_2$ID_PERSONA),]

registres_new$FORMACIONS_TRIM <- registres_new_2$N_FORMACIONS_2






# NUMERO D'ASSESSORAMENTS

assessorament$DALTA <- as.Date(assessorament$INICI, format = "%d/%m/%Y")
assessorament$INICI <- as.Date(assessorament$DALTA, format = "%d/%m/%y")
assessorament <- dplyr::select(assessorament, -DALTA)

registres_new_2 <- registres_new %>%
  left_join(assessorament, by = "ID_PERSONA") %>%
  filter(!is.na(INICI) & INICI >= DALTA & (is.na(FIN_TRIMESTRE) | INICI <= FIN_TRIMESTRE)) %>%
  group_by(ID_PERSONA, DALTA, FIN_TRIMESTRE) %>%
  summarize(N_ASSESSORAMENTS_2 = n(), .groups = "drop") %>%
  right_join(registres_new, by = c("ID_PERSONA", "DALTA", "FIN_TRIMESTRE")) %>%
  select(ID_PERSONA, DALTA, FIN_TRIMESTRE, ID, N_ASSESSORAMENTS_2)

registres_new_2<-registres_new_2[order(registres_new_2$ID_PERSONA),]
registres_new$ASSESSORAMENTS_TRIM <- registres_new_2$N_ASSESSORAMENTS_2


registres_new$FORMACIONS_TRIM [is.na(registres_new$FORMACIONS_TRIM )] <- 0
registres_new$ASSESSORAMENTS_TRIM [is.na(registres_new$ASSESSORAMENTS_TRIM )] <- 0
registres_new$N_FORMACIONS [is.na(registres_new$N_FORMACIONS )] <- 0
registres_new$N_ASSESSORAMENTS [is.na(registres_new$N_ASSESSORAMENTS )] <- 0


registres_new$ACTIVITAT <- registres_new$FORMACIONS_TRIM + registres_new$ASSESSORAMENTS_TRIM
registres_new$ACTIVITAT_TOTAL <- registres_new$N_FORMACIONS + registres_new$N_ASSESSORAMENTS
registres_new$PERCENTATGE_ACT<- registres_new$ACTIVITAT/registres_new$ACTIVITAT_TOTAL
registres_new$PERCENTATGE_ACT [registres_new$PERCENTATGE_ACT >1 ] <- 1




# n_any no es rellevant
# Proposta: NN recurent podriem fer una xarxa amb una serie temporal però necesitariam més informació a cada instant de l'afiliació
# Algoritme molt més complex (necesita molt entrenament)
# A més falta info


# crear més files, una per trimestre d'afiliacio y variale resposta trim seguent afiliat o no
# acumulat, total afiliacio (percentatge)



# Els casos antics (ja donats de baixa )



# PRIMER MODEL INPUT


library(lubridate)
library(tidyverse)
library(dplyr)

# Selecionem els usuaris que es van afiliar entre el gener i el març de 2022

assessorament =assessorament %>% mutate(INICI = DALTA)
assessorament =assessorament %>% select(-'DALTA')


#Registres amb assessorament amb DALTA entre dates

assessorament$INICI<- as.Date(assessorament$INICI, "%d/%m/%Y")
concicio_a<- assessorament$INICI > as.Date('2021/01/01') &
              assessorament$INICI < as.Date('2021/09/01')
assessorament_f <-  assessorament[concicio_a,]

condicio_r = registres$ID_PERSONA %in% assessorament_f$ID_PERSONA &
              registres$DALTA> as.Date('2021/01/01')&
              registres$DALTA< as.Date('2021/03/31') 
registres_f <- registres[condicio_r,]


# Taula frequencia assessoraments:

frecuencia <- table(registres_f$ID_PERSONA)
frecuencia_df <- as.data.frame(frecuencia)
colnames(frecuencia_df) <- c( "ID_PERSONA", "Frecuencia")

#Registres amb FORMACIO amb DALTA entre dates

formacio<- na.omit(formacio, cols = "INICI")
formacio$INICI<- as.Date(formacio$INICI, "%d/%m/%Y")
concicio_a<- formacio$INICI > as.Date('2021/01/01') &
  formacio$INICI < as.Date('2021/09/30')
formacio_f <-  formacio[concicio_a,]


condicio_r = registres$ID_PERSONA %in% formacio_f$ID_PERSONA &
  registres$DALTA> as.Date('2021/01/01')&
  registres$DALTA< as.Date('2021/03/31') 
registres_f <- registres[condicio_r,]


# Taula frequencia assessoraments:

frecuencia <- table(registres_f$ID_PERSONA)
frecuencia2_df <- as.data.frame(frecuencia)
colnames(frecuencia2_df) <- c( "ID_PERSONA", "Frecuencia")


# Creeem input

input <- registres %>% filter(DALTA > as.Date('2021/01/01') & DALTA < as.Date('2021/03/31'))
input$DFI <- as.Date('2021/09/31')

input <- select(input, c(-'N_FORMACIONS' , -'N_ASSESSORAMENTS'))

input$N_ASSESSORAMENTS = frecuencia_df$Frecuencia[match(input$ID_PERSONA, frecuencia_df$ID_PERSONA)]
input$N_FORMACIONS = frecuencia2_df$Frecuencia[match(input$ID_PERSONA, frecuencia2_df$ID_PERSONA)]

input$N_FORMACIONS<- ifelse(is.na(input$N_FORMACIONS), 0, input$N_FORMACIONS)
input$N_ASSESSORAMENTS<- ifelse(is.na(input$N_ASSESSORAMENTS), 0, input$N_ASSESSORAMENTS)

table(input$CENTRE_NOMINA)
table(input$SECTOR)

input <- input%>% select(c("ID", "ID_PERSONA", "M_ALTA", "CENTRE_ELECCIONS", "N_FORMACIONS", "N_ASSESSORAMENTS", "SLABORAL"))


# VARIABLE TARGET: Es donarà de baixa del gener al març de 2022?)
registres$DBAIXA<- as.Date(registres$DBAIXA, "%Y/%m/%d")

# Persones que es donaran de baixa:

condicio_r = registres$ID_PERSONA %in% input$ID_PERSONA &
  registres$DBAIXA> as.Date('2021/01/01')&
  registres$DBAIXA< as.Date('2021/09/30') 
registres_f <- registres[condicio_r,]

input$BAIXA_AFILIACIO<-ifelse(input$ID_PERSONA %in% registres_f$ID_PERSONA, 1, 0)

input$NUM_AFILIACIO <- as.numeric(substr(input$ID, nchar(input$ID), nchar(input$ID)))

persones_f <- persones%>% filter(ID_PERSONA %in% input$ID_PERSONA)
persones_f$DNAIXEMENT <- as.Date(persones_f$DNAIXEMENT, "%d/%m/%Y")

persones_f <- persones_f%>% select(c(ID_PERSONA, DNAIXEMENT))
input_2<- merge(input, persones_f, by= "ID_PERSONA", all.x = TRUE)
input_2$EDAT <- as.integer(difftime(as.Date('2021/03/31'), input_2$DNAIXEMENT, units = "days") / 365)
input_2 <- input_2%>% select(-DNAIXEMENT)


input_2 %>% names()


write.csv(input_2, "input_2.csv")

input_3<-read.csv("input_2.csv")

input_3%>% names()









