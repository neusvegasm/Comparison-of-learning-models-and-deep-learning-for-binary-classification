library(lubridate)



## Importar taules

setwd("~/UAB/4t/TFG/DadesEstudiAfil")

acta <- read.delim('acta.txt', sep = ';')
afiliacio <- read.delim('afiliacio.txt', sep = ';')
agteess <- read.delim('agcteess.txt', sep = ';')
assessorament <- read.delim('assessorament.txt', sep = ';')
baixa <- read.delim('baixa.txt', sep = ';')
candidat <- read.delim('candidat.txt', sep = ';')
candidatura <- read.delim('candtura.txt', sep = ';')
categories <- read.delim('categories.txt', sep = ';')
causa <- read.delim('causa.txt', sep = ';')
causa2 <- read.delim('causa2.txt', sep = ';')
cnaes <- read.delim('cnaes.txt', sep = ';')
colegi <- read.delim('colegi.txt', sep = ';')
comarques <- read.delim('comarques.txt', sep = ';')
convenis <- read.delim('convenis.txt', sep = ';')
cotitzacions <- read.delim('cotitzacions.txt', sep = ';')
empreses <- read.delim('empreses.txt', sep = ';')
formacio <- read.delim('formacio.txt', sep = ';')
localitats <- read.delim('localitats.txt', sep = ';')
materies <- read.delim('materies.txt', sep = ';')
motius <- read.delim('motius.txt', sep = ';')
origen <- read.delim('origen.txt', sep = ';')
persones <- read.delim('persones.txt', sep = ';')
preavis <- read.delim('preavis.txt', sep = ';')
rams <- read.delim('rams.txt', sep = ';')
relagreess <- read.delim('relagreess.txt', sep = ';')
resultats <- read.delim('resultats.txt', sep = ';')
sectors <- read.delim('sectors.txt', sep = ';')
serveis_gtj <- read.delim('serveis_gtj.txt', sep = ';')
sigles <- read.delim('sigles.txt', sep = ';')
situacio_laboral <- read.delim('situacio_laboral.txt', sep = ';')
submotius <- read.delim('submotius.txt', sep = ';')
tipus_afiliacio <- read.delim('tipus_afiliacio.txt', sep = ';')
tipus_contracte <- read.delim('tipus_contracte.txt', sep = ';')
tipus_pagament <- read.delim('tipus_pagament.txt', sep = ';')
unions_intercomarcals <- read.delim('unions_intercomarcals.txt', sep = ';')
unions_locals <- read.delim('unions_locals.txt', sep = ';')
#provincies <- read.delim('provincies.txt', sep = ';')


#Constuïm df

id_persona <- unique(persones$ID_PERSONA)

#Persones que estan actualment donats d'alta
afiliats_actual<-afiliacio[afiliacio$DBAIXA=="", 
              c('ID_PERSONA', 'CONTRACTE_AFIL', 'SLABORAL', 'CATEG', 'ORIGEN', 'TIPAF',
                'TIPPAG', 'TIPPAG_PERIODO', 'DALTA')]

any(table(id_persona)>1) #check: cap id repetit
all(afiliats_actual$ID_PERSONA %in% id_persona)#check: tots els afiliats es troben al llistat de persones

#FUNCIO: diferencia en mesos:
elapsed_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}


afiliats_actual$DALTA<-as.Date(afiliats_actual$DALTA, "%d/%m/%Y")
afiliats_actual[afiliats_actual$DALTA>as.Date("01/01/2005", "%d/%m/%Y"),]

afiliacio[afiliacio$DALTA>as.Date("01/01/2005", "%d/%m/%Y"),]

hist(elapsed_months(Sys.Date() , afiliats_actual$DALTA), breaks = 400, xlim = c(0,12))
which.max(afiliats_actual$DALTA)
afiliats_actual[66179,]
afiliacio[afiliacio$DALTA="01/07/2023",]



baixa$DALTA<-as.Date(baixa$DALTA, "%d/%m/%Y")

max(baixa$DBAIXA)


# plot months - sum baixes

plot_data<-data.frame(table(baixa$DBAIXA))
names(plot_data)<- c('date', 'baixes')




#create time series plot
plot_data$date<- as.Date(plot_data$date, "%d/%m/%Y")
str(plot_data)

plot_data<-filter(plot_data ,plot_data$date > as.Date('01/01/2005', "%d/%m/%Y"))

library(ggplot2)
library(dplyr)

p <- ggplot(plot_data, aes(x=date, y=baixes)) +
  geom_line() + 
  xlab("") +
  scale_x_date(date_labels = "%m-%Y")

p




# Càlcul de num mesos que porten afiliats:



## Preprocesament de dades

# Talls temporals mesos o trimestres
# Segons la quota, rangs de salaris (només en pagaments mensuals)


# Persones: ID, gènere, localitat
# Empreses: sector, d'ata d'alta, data de baixa (Quan es crea l'empresa i ) (conveni d’aplicació, conveni subsidiari.??)
# Afiliats: tipus d’afiliació, tipus de pagament, data d’alta, data de la última baixa (cada persona 1 vegada, data de baixa NA si no s'ha donat mai de baixa) (tipus de treballador)
# Baixes: tipus d’afiliació, tipus de pagament, data d’alta, data baixa, codi afiliat confe ( historic )
# Cotitzacions: any, mes ,import a pagar, import pagat, data pagament, forma de pagament (rang salarial no possible)(forma de pagament pt anar variant)
# Preavisos: ???
# Acta: ???
# Candidatura: ?? Les empreses poden tenir o no eleccions sindicals
# Candidats: número ordre a la llista, dni, sigla, antiguitat a l’empresa, si té data d'alta vol dir que ha estat escollit com a delegat
#(molts delegats 45% no són afiliats) ( candidats per NIF i per provincia)
# Agrupa_emp: Codi de l’agrupació, codi de centre de treball relacionat, indicador d’estat?
# Agctss:??
# Segsoc: número de treballadors/res de l'empresa (NIF)
# Visites: dia/es-hora/es i lloc de l’esdeveniment, àmbit de l’esdeveniment, organització convocant, agent/s sindicals, persones convocades. (NO HI HA INFO ACTUALITZADA)
# Ceres_convenis: ???
# Usuari/client (assessorament):  codi_visita,  data alta, situació afiliació, mesos afiliat
# Assessors/advocats: Codi assessor, dni, data alta? data baixa?, referencia organitzativa
# Visites: data alta?, data baixa?, resultat
# Contraris: ??
# Expedient: data alta, materia, tipus servei,dni client, dni advocat, dades facturació
# SECCIÓ ESTRUCTURA SINDICAL Sí
# Alumnes: situació laboral, data naixement, estudis, 
# Cursos: Identificador de l’acció, descripció, tipus, modalitat presencialitat, hores lectives, número alumnes
# Formadors: Els formadors són també afiliats? 





## Possibles variables:

# Variable resposta: Afilitat trimestre (o any) següent: Sí/No
# Nombre de mesos afiliat(actual)
# Mesos afiliat (màxim per persona)
# Mesos afiliat (mínim per persona)
# Tipus de pagament
# Forma de pagament
# Candidat (Sí/no)
# Sector empresa
# Localitat / província
# Visita d'assessorament (sí/no)
# Número de visites
# Mesos d'afiliació abans de la primera visita
# Mesos d'afiliació després de la última visita
# Número de treballadors de l'empresa
# Àmbit d'esdeveniment, visites
# Formació (Sí/no)
# Matèria de la formació 
# Matèria del curs
