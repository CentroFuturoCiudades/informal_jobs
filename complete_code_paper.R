rm(list=ls())


#Loading required libraries
library(tidyverse)
library(randomForest)
library(caret)



# Creation of ENOE base ---------------------------------------------------

tamaño_hogar <- read_csv("enoe_juntado4t19.csv")  %>% 
  mutate(hog_id = paste0( CD_A,"_", ENT, "_", CON, "_", V_SEL )) %>% 
  group_by(hog_id) %>% 
  summarise(habitantes = n())


enoe <- read_csv("enoe_juntado4t19.csv") %>% 
  filter(ENT %in% "19")



enoe_b <- enoe %>% 
  mutate(hog_id = paste0( CD_A,"_", ENT, "_", CON, "_", V_SEL))  %>% 
  filter(P1.coe1 == 1) %>%  ## Se filtra por quien sí trabajo la semana pasada
  left_join(tamaño_hogar, by = "hog_id") %>% 
  transmute(
    genero = case_when(
      SEX %in% "1" ~ "H",
      SEX %in% "2" ~ "F",
      TRUE ~ NA
    ), 
    ocupacion = case_when(
      POS_OCU %in% "1" ~ "trabajador", 
      POS_OCU %in% "2" ~ "trabajador", 
      POS_OCU %in% "3" ~ "independiente",
      TRUE ~ "otro"
    ),
    sector = case_when(
      SCIAN %in% "1" ~ "Agricultura y ganadería",
      SCIAN %in% "2" ~ "Minería",
      SCIAN %in% "3" ~ "Otro",
      SCIAN %in% "4" ~ "Construcción",
      SCIAN %in% "5" ~ "Industria manufacturera",
      SCIAN %in% "6" ~ "Comercio",
      SCIAN %in% "7" ~ "Comercio",
      SCIAN %in% "8" ~ "Transporte y comunicaciones",
      SCIAN %in% "9" ~ "Otro",
      SCIAN %in% "10" ~ "Servicios",
      SCIAN %in% "11" ~ "Servicios",
      SCIAN %in% "12" ~ "Servicios",
      SCIAN %in% "13" ~ "Servicios",
      SCIAN %in% "14" ~ "Servicios",
      SCIAN %in% "15" ~ "Servicios",
      SCIAN %in% "16" ~ "Servicios",
      SCIAN %in% "17" ~ "Servicios",
      SCIAN %in% "18" ~ "Servicios",
      SCIAN %in% "19" ~ "Servicios",
      SCIAN %in% "20" ~ "Gobierno",
      SCIAN %in% "21" ~ "Otro",
    ), 
    edad = case_when(
      EDA < 3 ~ "0-2",
      EDA < 5 ~ "3-4",
      EDA < 6 ~ "5",
      EDA < 8 ~ "6-7",
      EDA < 12 ~ "8-11",
      EDA < 15 ~ "12-14",
      EDA < 18 ~ "15-17",
      EDA < 25 ~ "18-24",
      EDA < 50 ~ "25-49",
      EDA < 60 ~ "50-59",
      EDA < 65 ~ "60-64",
      EDA < 135 ~ "65-130",
      TRUE ~ "Unknown"
    ), 
    escolaridad = case_when(
      CS_P13_1 %in% "0" ~ "Sin Instrucción",
      CS_P13_1 %in% "1" ~ "Sin Instrucción",
      CS_P13_1 %in% "2" ~ "Primaria o Secundaria",
      CS_P13_1 %in% "3" ~ "Primaria o Secundaria",
      CS_P13_1 %in% "4" ~ "Carrera técnica o preparatoria",
      CS_P13_1 %in% "5" ~ "Carrera técnica o preparatoria",
      CS_P13_1 %in% "6" ~ "Carrera técnica o preparatoria",
      CS_P13_1 %in% "7" ~ "Licenciatura",
      CS_P13_1 %in% "8" ~ "Postgrado",
      CS_P13_1 %in% "9" ~ "Postgrado",
      CS_P13_1 %in% "99" ~ "Otro"
    ),
    informal = case_when(
      EMP_PPAL == 1 ~ 1, ## si es informal va a ser 1
      EMP_PPAL == 2 ~ 0,
      EMP_PPAL == 0 ~ 100,
    ),
    municipio = case_when(
      MUN %in% "1" ~ "abasolo", 
      MUN %in% "6" ~ "apodaca", 
      MUN %in% "9" ~ "cadereyta",
      MUN %in% "12" ~ "flores", 
      MUN %in% "10" ~ "carmen", 
      MUN %in% "18" ~ "garcia", 
      MUN %in% "21" ~ "escobedo", 
      MUN %in% "25" ~ "zuazua", 
      MUN %in% "26" ~ "guadalupe", 
      MUN %in% "47" ~ "hidalgo",
      MUN %in% "31" ~ "juarez", 
      MUN %in% "39" ~ "monterrey",
      MUN %in% "41" ~ "pesqueria", 
      MUN %in% "45" ~ "salinas", 
      MUN %in% "46" ~ "san_nicolas", 
      MUN %in% "19" ~ "san_pedro", 
      MUN %in% "48" ~ "santa_catarina", 
      MUN %in% "49" ~ "santiago", 
      TRUE ~ "otro"
    ), factor = FAC
  ) %>% 
  filter(!informal == 100, 
         !escolaridad %in% "Otro"
  )  %>% 
  mutate(informal = as.factor(informal)) 



# Creation of OD survey ---------------------------------------------------

od <- read_csv("base_eodh/datos_limpios_tiempos.csv")  %>% 
  filter(Motivo %in% "trabajo") %>%  ## Filtro para los viajes que son de trabajo
  distinct(`H-P`, .keep_all = T)  %>% 
  mutate(
    genero = genero, 
    edad = case_when(
      Edad < 3 ~ "0-2",
      Edad < 5 ~ "3-4",
      Edad < 6 ~ "5",
      Edad < 8 ~ "6-7",
      Edad < 12 ~ "8-11",
      Edad < 15 ~ "12-14",
      Edad < 18 ~ "15-17",
      Edad < 25 ~ "18-24",
      Edad < 50 ~ "25-49",
      Edad < 60 ~ "50-59",
      Edad < 65 ~ "60-64",
      Edad < 135 ~ "65-130",
      TRUE ~ "Unknown"
    ) , ### modificar edad para que sea igual que en el censo, probar efectos en el modelo
    escolaridad = case_when(
      Estudios %in% "Primaria o secundaria"~ "Primaria o Secundaria",
      TRUE ~ Estudios
    ) , 
    municipio = case_when(
      Cod_MunDomicilio %in% "Abasolo" ~ "otro", 
      Cod_MunDomicilio %in% "Apodaca" ~ "apodaca", 
      Cod_MunDomicilio %in% "Cadereyta Jiménez" ~ "cadereyta",
      Cod_MunDomicilio %in% "Ciénega de Flores" ~ "flores", 
      Cod_MunDomicilio %in% "El Carmen" ~ "carmen", 
      Cod_MunDomicilio %in% "García" ~ "garcia", 
      Cod_MunDomicilio %in% "General Escobedo" ~ "escobedo", 
      Cod_MunDomicilio %in% "General Zuazua" ~ "zuazua", 
      Cod_MunDomicilio %in% "Guadalupe" ~ "guadalupe", 
      Cod_MunDomicilio %in% "Hidalgo" ~ "hidalgo",
      Cod_MunDomicilio %in% "Juárez" ~ "juarez", 
      Cod_MunDomicilio %in% "Monterrey" ~ "monterrey",
      Cod_MunDomicilio %in% "Pesquería" ~ "otro", 
      Cod_MunDomicilio %in% "Salinas Victoria" ~ "salinas", 
      Cod_MunDomicilio %in% "San Nicolás de los Garza" ~ "san_nicolas", 
      Cod_MunDomicilio %in% "San Pedro Garza García" ~ "san_pedro", 
      Cod_MunDomicilio %in% "Santa Catarina" ~ "santa_catarina", 
      Cod_MunDomicilio %in% "Santiago" ~ "santiago", 
      TRUE ~ "otro"
    ),
    ocupacion = case_when(
      Ocupacion %in% c("Empleado (a)", "Obrero(a)", "Profesionista empleado", "Comerciante") ~ "trabajador",
      Ocupacion %in% c("Profesionista independiente", "Profesionista Independiente") ~ "independiente",
      TRUE ~ "otro"
    ),
    sector =  case_when(
      !SectorEconom %in% "Otro" ~ SectorEconom,
      SectorEconom_O %in% "Ayudante general"  ~  "Servicios",
      SectorEconom_O %in% "Pemex"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Refineria"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Salud"  ~  "Servicios",
      SectorEconom_O %in% "Hacen Chocolates"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Educación"  ~  "Servicios",
      SectorEconom_O %in% "FABRICA"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Recolector de metal"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Albañil"  ~  "Construcción",
      SectorEconom_O %in% "Industria Alimenticia"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Soldador"  ~  "Construcción",
      SectorEconom_O %in% "Privado"  ~  "Servicios",
      SectorEconom_O %in% "Alimentos"  ~  "Servicios",
      SectorEconom_O %in% "Guardia de seguridad"  ~  "Servicios",
      SectorEconom_O %in% "Maquila"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Seguridad"  ~  "Servicios",
      SectorEconom_O %in% "Educacion"  ~  "Servicios",
      SectorEconom_O %in% "Energeticos"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Fabrica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Industria Metalúrgica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Maestra"  ~  "Servicios",
      SectorEconom_O %in% "Maestro"  ~  "Servicios",
      SectorEconom_O %in% "Mantenimiento"  ~  "Servicios",
      SectorEconom_O %in% "Mecánico"  ~  "Servicios",
      SectorEconom_O %in% "Operaria"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Vigilante"  ~  "Servicios",
      SectorEconom_O %in% "Carpintero"  ~  "Construcción",
      SectorEconom_O %in% "Doctor"  ~  "Servicios",
      SectorEconom_O %in% "Guarderia pemex"  ~  "Servicios",
      SectorEconom_O %in% "Guardia"  ~  "Servicios",
      SectorEconom_O %in% "INDUSTRIA ALIMENTICIA"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Independiente"  ~  "Servicios",
      SectorEconom_O %in% "Limpieza"  ~  "Servicios",
      SectorEconom_O %in% "MECANICO"  ~  "Industria manufacturera",
      SectorEconom_O %in% "PRIVADO"  ~  "Servicios",
      SectorEconom_O %in% "Repartidor"  ~  "Servicios",
      SectorEconom_O %in% "TALLER"  ~  "Servicios",
      SectorEconom_O %in% "ALIMENTOS"  ~  "Servicios",
      SectorEconom_O %in% "Carpintería"  ~  "Construcción",
      SectorEconom_O %in% "EDUCACIÓN"  ~  "Servicios",
      SectorEconom_O %in% "Entretenimiento"  ~  "Servicios",
      SectorEconom_O %in% "Fábrica de chocolate"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Gasolinero"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Imss"  ~  "Servicios",
      SectorEconom_O %in% "Jardinero"  ~  "Servicios",
      SectorEconom_O %in% "MECANICA"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Montacargista"  ~  "Servicios",
      SectorEconom_O %in% "Obrero"  ~  "Construcción",
      SectorEconom_O %in% "Por su cuenta"  ~  "Servicios",
      SectorEconom_O %in% "SALUD"  ~  "Servicios",
      SectorEconom_O %in% "Ventas"  ~  "Comercio",
      SectorEconom_O %in% "educacion"  ~  "Servicios",
      SectorEconom_O %in% "AUTOMOTRIZ"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Abogado"  ~  "Servicios",
      SectorEconom_O %in% "Alimenticia"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Almacenista"  ~  "Comercio",
      SectorEconom_O %in% "Arenas silicas"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Banco"  ~  "Servicios",
      SectorEconom_O %in% "CFE"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Chofer"  ~  "Servicios",
      SectorEconom_O %in% "Cocinera"  ~  "Servicios",
      SectorEconom_O %in% "Costurera"  ~  "Servicios",
      SectorEconom_O %in% "Doméstica"  ~  "Servicios",
      SectorEconom_O %in% "Educación privada"  ~  "Servicios",
      SectorEconom_O %in% "Elaboración de tamales"  ~  "Servicios",
      SectorEconom_O %in% "Enfermera"  ~  "Servicios",
      SectorEconom_O %in% "Ferreteria"  ~  "Comercio",
      SectorEconom_O %in% "Hospital de pemex"  ~  "Servicios",
      SectorEconom_O %in% "Industria Gastronómica"  ~  "Servicios",
      SectorEconom_O %in% "Intendente"  ~  "Servicios",
      SectorEconom_O %in% "LIMPIEZA"  ~  "Servicios",
      SectorEconom_O %in% "Limpieza en casas"  ~  "Servicios",
      SectorEconom_O %in% "Manufactura"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Metal mecánica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Montacarguista"  ~  "Servicios",
      SectorEconom_O %in% "Negocio propio"  ~  "Servicios",
      SectorEconom_O %in% "OPERARIA"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Operario"  ~  "Industria manufacturera",
      SectorEconom_O %in% "RECICLADORA"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Recolección de residuos"  ~  "Servicios",
      SectorEconom_O %in% "Sector salud"  ~  "Servicios",
      SectorEconom_O %in% "Supervisor"  ~  "Servicios",
      SectorEconom_O %in% "Tacos"  ~  "Servicios",
      SectorEconom_O %in% "Velador"  ~  "Servicios",
      SectorEconom_O %in% "Vende tacos"  ~  "Comercio",
      SectorEconom_O %in% "industria alimenticia"  ~  "Industria manufacturera",
      SectorEconom_O %in% "recolector de metal"  ~  "Comercio",
      SectorEconom_O %in% "ABOGADA"  ~  "Servicios",
      SectorEconom_O %in% "ACEROS"  ~  "Industria manufacturera",
      SectorEconom_O %in% "ADMINISTRACIO N"  ~  "Servicios",
      SectorEconom_O %in% "Abastecimiento industrial"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Aceros"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Administracion"  ~  "Servicios",
      SectorEconom_O %in% "Administración de proyectos"  ~  "Servicios",
      SectorEconom_O %in% "Administrativo"  ~  "Servicios",
      SectorEconom_O %in% "Afanador"  ~  "Servicios",
      SectorEconom_O %in% "Alimentación y bebidas"  ~  "Servicios",
      SectorEconom_O %in% "Almacen"  ~  "Comercio",
      SectorEconom_O %in% "Artesanias"  ~  "Comercio",
      SectorEconom_O %in% "Artesano"  ~  "Comercio",
      SectorEconom_O %in% "Asesor"  ~  "Servicios",
      SectorEconom_O %in% "Atnc cliente"  ~  "Servicios",
      SectorEconom_O %in% "Autopartes"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Ayudante"  ~  "Construcción",
      SectorEconom_O %in% "BANCARIO"  ~  "Servicios",
      SectorEconom_O %in% "Bienes raices"  ~  "Comercio",
      SectorEconom_O %in% "Bodega"  ~  "Comercio",
      SectorEconom_O %in% "CAJERA"  ~  "Servicios",
      SectorEconom_O %in% "CALL CENTER"  ~  "Servicios",
      SectorEconom_O %in% "CARNICERIA"  ~  "Servicios",
      SectorEconom_O %in% "CARPINTERIA"  ~  "Servicios",
      SectorEconom_O %in% "CERREJERIA"  ~  "Servicios",
      SectorEconom_O %in% "CHOFER PARTICULAR"  ~  "Servicios",
      SectorEconom_O %in% "CLINICA"  ~  "Servicios",
      SectorEconom_O %in% "COLEGIO"  ~  "Servicios",
      SectorEconom_O %in% "COMIDAS"  ~  "Servicios",
      SectorEconom_O %in% "COMUNICACION"  ~  "Servicios",
      SectorEconom_O %in% "Campo de golf"  ~  "Servicios",
      SectorEconom_O %in% "Carbonera"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Cargador"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Carnes empaque"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Carnicero"  ~  "Comercio",
      SectorEconom_O %in% "Casa"  ~  "Servicios",
      SectorEconom_O %in% "Catedrático Universidad"  ~  "Servicios",
      SectorEconom_O %in% "Central de autobuses"  ~  "Servicios",
      SectorEconom_O %in% "Centro de a tensión a clientes telefónica"  ~  "Servicios",
      SectorEconom_O %in% "Chófer en casa"  ~  "Servicios",
      SectorEconom_O %in% "Comercio ambulante"  ~  "Comercio",
      SectorEconom_O %in% "Compañía Everest"  ~  "Servicios",
      SectorEconom_O %in% "Compañía dentro de pemex"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Compañías interior de refinería"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Compra"  ~  "Comercio",
      SectorEconom_O %in% "Compra fierro metal"  ~  "Comercio",
      SectorEconom_O %in% "Comunicacion"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Contadora"  ~  "Servicios",
      SectorEconom_O %in% "Costuras"  ~  "Servicios",
      SectorEconom_O %in% "Cuida niños"  ~  "Servicios",
      SectorEconom_O %in% "Cuida rancho"  ~  "Servicios",
      SectorEconom_O %in% "Cuida su ganado"  ~  "Agricultura y ganadería",
      SectorEconom_O %in% "Cuida una quinta"  ~  "Servicios",
      SectorEconom_O %in% "DEPOSITO DE CERVEZA"  ~  "Comercio",
      SectorEconom_O %in% "DISEÑADOR GRAFICO"  ~  "Servicios",
      SectorEconom_O %in% "DISEÑO INDUSTRIAL"  ~  "Servicios",
      SectorEconom_O %in% "Da clases de beisbol"  ~  "Servicios",
      SectorEconom_O %in% "Dentista"  ~  "Servicios",
      SectorEconom_O %in% "Despacho"  ~  "Servicios",
      SectorEconom_O %in% "Director"  ~  "Comercio",
      SectorEconom_O %in% "Diseño"  ~  "Servicios",
      SectorEconom_O %in% "Docente"  ~  "Servicios",
      SectorEconom_O %in% "Domestico"  ~  "Servicios",
      SectorEconom_O %in% "EDUCATIVO"  ~  "Servicios",
      SectorEconom_O %in% "ELECTRICIDAD Y PLOMERIA"  ~  "Industria manufacturera",
      SectorEconom_O %in% "EMPLEADO"  ~  "Comercio",
      SectorEconom_O %in% "ENFERMERIA"  ~  "Servicios",
      SectorEconom_O %in% "ES VENDEDOR EN EL MERCADO DE ABASTOS DE SAN NICOLAS"  ~  "Comercio",
      SectorEconom_O %in% "ESCUELA"  ~  "Servicios",
      SectorEconom_O %in% "ESCUELA GUARDIA"  ~  "Servicios",
      SectorEconom_O %in% "ESTETICA"  ~  "Servicios",
      SectorEconom_O %in% "Educación Privada"  ~  "Servicios",
      SectorEconom_O %in% "Educativo"  ~  "Servicios",
      SectorEconom_O %in% "Empacadora"  ~  "Comercio",
      SectorEconom_O %in% "Empacadora de carnes"  ~  "Comercio",
      SectorEconom_O %in% "Empaque de carne"  ~  "Comercio",
      SectorEconom_O %in% "Empleada domestica"  ~  "Servicios",
      SectorEconom_O %in% "Empleada en un abarrotes"  ~  "Comercio",
      SectorEconom_O %in% "Empleado"  ~  "Servicios",
      SectorEconom_O %in% "Empleeado en un local"  ~  "Comercio",
      SectorEconom_O %in% "Empresa"  ~  "Servicios",
      SectorEconom_O %in% "Empresa que se dedica a empaque"  ~  "Industria manufacturera",
      SectorEconom_O %in% "En fábrica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "En los Medios de comunicación"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Enfermero"  ~  "Servicios",
      SectorEconom_O %in% "Entrenador de deportes"  ~  "Servicios",
      SectorEconom_O %in% "Es demostradora"  ~  "Servicios",
      SectorEconom_O %in% "Escuela Religiosa"  ~  "Servicios",
      SectorEconom_O %in% "Estilista"  ~  "Servicios",
      SectorEconom_O %in% "Eventual (particular)"  ~  "Servicios",
      SectorEconom_O %in% "FABRICA ENSAMBLADOR"  ~  "Industria manufacturera",
      SectorEconom_O %in% "FABRICA OFICINA"  ~  "Servicios",
      SectorEconom_O %in% "FIESTAS"  ~  "Servicios",
      SectorEconom_O %in% "FUERZ CIVIL"  ~  "Gobierno",
      SectorEconom_O %in% "Fabrica de chocolate"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Fabrica de escobas"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Fabrica ensamble de carros kia"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Fabricación de plásticos"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Financiero"  ~  "Servicios",
      SectorEconom_O %in% "Frabrica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Fumigación"  ~  "Servicios",
      SectorEconom_O %in% "Fundidora cadereyta"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Fábrica de tu Unicel"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Gas"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Gasolineria"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Gimnasia"  ~  "Servicios",
      SectorEconom_O %in% "Grupo energeticos planta 2"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Guarderia"  ~  "Servicios",
      SectorEconom_O %in% "Guarderia de pemex"  ~  "Servicios",
      SectorEconom_O %in% "Guardia Refineria"  ~  "Servicios",
      SectorEconom_O %in% "HOSPITAL"  ~  "Servicios",
      SectorEconom_O %in% "HOTEL"  ~  "Servicios",
      SectorEconom_O %in% "Hace tortillas de harina"  ~  "Servicios",
      SectorEconom_O %in% "Hacen pinturas"  ~  "Servicios",
      SectorEconom_O %in% "Herrero"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Hotel"  ~  "Servicios",
      SectorEconom_O %in% "IMPRENTA"  ~  "Servicios",
      SectorEconom_O %in% "INSPECTOR"  ~  "Servicios",
      SectorEconom_O %in% "INSTALACION DE FIBRA OPTICA"  ~  "Servicios",
      SectorEconom_O %in% "INTENDENCIA"  ~  "Servicios",
      SectorEconom_O %in% "Ind papelera"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Industria Alimentaria"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Industria Farmacéutica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Industria Quimica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Iniciativa privada"  ~  "Servicios",
      SectorEconom_O %in% "Intendente de kinder"  ~  "Servicios",
      SectorEconom_O %in% "Investigacón"  ~  "Servicios",
      SectorEconom_O %in% "JARDINERIA"  ~  "Servicios",
      SectorEconom_O %in% "Jefe área"  ~  "Servicios",
      SectorEconom_O %in% "Jugeteria"  ~  "Comercio",
      SectorEconom_O %in% "La Costeña (chiles, fríjoles etc)"  ~  "Comercio",
      SectorEconom_O %in% "La reynera escobera"  ~  "Comercio",
      SectorEconom_O %in% "Labor solcial"  ~  "Servicios",
      SectorEconom_O %in% "Lava camiones"  ~  "Servicios",
      SectorEconom_O %in% "Lavador de camion"  ~  "Servicios",
      SectorEconom_O %in% "Lavador de camiones"  ~  "Servicios",
      SectorEconom_O %in% "Lic. en administración de empresas"  ~  "Servicios",
      SectorEconom_O %in% "Limpieza de casas"  ~  "Servicios",
      SectorEconom_O %in% "Limpieza de oficinas"  ~  "Servicios",
      SectorEconom_O %in% "Limpieza en casa"  ~  "Servicios",
      SectorEconom_O %in% "Local de artesanias"  ~  "Comercio",
      SectorEconom_O %in% "Local en los cavazos"  ~  "Comercio",
      SectorEconom_O %in% "Logística"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "MADEDERO"  ~  "Servicios",
      SectorEconom_O %in% "MAESTRA"  ~  "Servicios",
      SectorEconom_O %in% "MANTENIMIENTO"  ~  "Servicios",
      SectorEconom_O %in% "Maderera"  ~  "Comercio",
      SectorEconom_O %in% "Maesta"  ~  "Servicios",
      SectorEconom_O %in% "Maestra de pilates"  ~  "Servicios",
      SectorEconom_O %in% "Maestro de preparatoria"  ~  "Servicios",
      SectorEconom_O %in% "Mantenimiento técnico"  ~  "Servicios",
      SectorEconom_O %in% "Maquiladora"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Mecanico"  ~  "Servicios",
      SectorEconom_O %in% "Mecanico automotriz"  ~  "Servicios",
      SectorEconom_O %in% "Mecano"  ~  "Servicios",
      SectorEconom_O %in% "Medicina"  ~  "Servicios",
      SectorEconom_O %in% "Mercado"  ~  "Comercio",
      SectorEconom_O %in% "Mesera"  ~  "Servicios",
      SectorEconom_O %in% "Mesero"  ~  "Servicios",
      SectorEconom_O %in% "Mesero en Restaurante"  ~  "Servicios",
      SectorEconom_O %in% "Metalúrgica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Mozo"  ~  "Servicios",
      SectorEconom_O %in% "Muebles"  ~  "Comercio",
      SectorEconom_O %in% "Negocio Propio"  ~  "Comercio",
      SectorEconom_O %in% "Notificadora"  ~  "Servicios",
      SectorEconom_O %in% "OFICINA"  ~  "Servicios",
      SectorEconom_O %in% "Oficina"  ~  "Servicios",
      SectorEconom_O %in% "Oficinistas"  ~  "Servicios",
      SectorEconom_O %in% "Ojalatero"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Oxxo"  ~  "Servicios",
      SectorEconom_O %in% "PRESTAMOS HIPOTECARIA"  ~  "Comercio",
      SectorEconom_O %in% "Panadero"  ~  "Comercio",
      SectorEconom_O %in% "Pastelera"  ~  "Comercio",
      SectorEconom_O %in% "Personal"  ~  "Servicios",
      SectorEconom_O %in% "Pesca"  ~  "Agricultura y ganadería",
      SectorEconom_O %in% "Piedrera san angel"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Pinta casas"  ~  "Servicios",
      SectorEconom_O %in% "Prepa privada"  ~  "Servicios",
      SectorEconom_O %in% "Productor musical"  ~  "Servicios",
      SectorEconom_O %in% "Química"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Químico"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Químicos"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Reciclaje"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Reciclajes"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Recolección de metal"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Recolección de metales"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Refaccionista"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Refinería Pemex"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Refinería de pemex"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Rellenonsanitario"  ~  "Servicios",
      SectorEconom_O %in% "Repartidor en motocicleta"  ~  "Servicios",
      SectorEconom_O %in% "Restaurante"  ~  "Servicios",
      SectorEconom_O %in% "Restaurante de la familia"  ~  "Servicios",
      SectorEconom_O %in% "SALUD/FARMACIA"  ~  "Servicios",
      SectorEconom_O %in% "SEGURIDAD"  ~  "Servicios",
      SectorEconom_O %in% "Se hace. Chocolates"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Secretaria"  ~  "Servicios",
      SectorEconom_O %in% "Sector Salud"  ~  "Servicios",
      SectorEconom_O %in% "Separador de metales"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Siderurgica"  ~  "Industria manufacturera",
      SectorEconom_O %in% "TALER"  ~  "Servicios",
      SectorEconom_O %in% "TALLER SOLDADORA"  ~  "Industria manufacturera",
      SectorEconom_O %in% "TIENDA"  ~  "Comercio",
      SectorEconom_O %in% "TRABAJA EN UN PUESTO DE POLLOS"  ~  "Comercio",
      SectorEconom_O %in% "TRABAJO DOMESTICO"  ~  "Servicios",
      SectorEconom_O %in% "Tacos mañaneros"  ~  "Servicios",
      SectorEconom_O %in% "Taller"  ~  "Servicios",
      SectorEconom_O %in% "Taller Mecanico"  ~  "Servicios",
      SectorEconom_O %in% "Taller de Torno"  ~  "Servicios",
      SectorEconom_O %in% "Taller mecanico"  ~  "Servicios",
      SectorEconom_O %in% "Tapiceria"  ~  "Servicios",
      SectorEconom_O %in% "Taquero"  ~  "Servicios",
      SectorEconom_O %in% "Taxista"  ~  "Servicios",
      SectorEconom_O %in% "Tecnología"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Telmex"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Tesoreria de vialidad y transito"  ~  "Gobierno",
      SectorEconom_O %in% "Textiles"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Tienda de abarrotes"  ~  "Comercio",
      SectorEconom_O %in% "Tienda de artículos para el hoga"  ~  "Comercio",
      SectorEconom_O %in% "Tienda de ropa"  ~  "Comercio",
      SectorEconom_O %in% "Tornero"  ~  "Comercio",
      SectorEconom_O %in% "Tortilleria"  ~  "Comercio",
      SectorEconom_O %in% "Tortonero"  ~  "Comercio",
      SectorEconom_O %in% "Trabaja en su camion de carga"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "Trabaja en una pedrera"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Trabaja por su cuenta"  ~  "Servicios",
      SectorEconom_O %in% "Trabaja por su cuenta como Plomero"  ~  "Servicios",
      SectorEconom_O %in% "Trabajadora doméstica"  ~  "Servicios",
      SectorEconom_O %in% "Trabajan plástico"  ~  "Industria manufacturera",
      SectorEconom_O %in% "Trabajos en general por su cuenta limpia maleza, albañilería etc."  ~  "Servicios",
      SectorEconom_O %in% "Trabajos varios para el hogar"  ~  "Servicios",
      SectorEconom_O %in% "Técnico en refrigeración"  ~  "Servicios",
      SectorEconom_O %in% "Técnico mtto"  ~  "Servicios",
      SectorEconom_O %in% "Uber conductor"  ~  "Servicios",
      SectorEconom_O %in% "VENDE ROPA EN LOS MERCADOS"  ~  "Comercio",
      SectorEconom_O %in% "VENDEDOR"  ~  "Comercio",
      SectorEconom_O %in% "Vende cartón y plastico"  ~  "Comercio",
      SectorEconom_O %in% "Vende comida"  ~  "Comercio",
      SectorEconom_O %in% "Vende fruta preparada"  ~  "Comercio",
      SectorEconom_O %in% "Vendedora"  ~  "Comercio",
      SectorEconom_O %in% "Vendedores"  ~  "Comercio",
      SectorEconom_O %in% "Venta de ropa"  ~  "Comercio",
      SectorEconom_O %in% "Viajes"  ~  "Servicios",
      SectorEconom_O %in% "Vidriería"  ~  "Comercio",
      SectorEconom_O %in% "Zapateria"  ~  "Comercio",
      SectorEconom_O %in% "asesora de ventas"  ~  "Servicios",
      SectorEconom_O %in% "cobrador"  ~  "Comercio",
      SectorEconom_O %in% "cuenta propia"  ~  "Servicios",
      SectorEconom_O %in% "empresa"  ~  "Servicios",
      SectorEconom_O %in% "industria Refresquera"  ~  "Industria manufacturera",
      SectorEconom_O %in% "jardineria"  ~  "Servicios",
      SectorEconom_O %in% "maestro"  ~  "Servicios",
      SectorEconom_O %in% "municipio"  ~  "Gobierno",
      SectorEconom_O %in% "oficinista"  ~  "Servicios",
      SectorEconom_O %in% "privado"  ~  "Servicios",
      SectorEconom_O %in% "restaurante-alimentos"  ~  "Servicios",
      SectorEconom_O %in% "salud"  ~  "Servicios",
      SectorEconom_O %in% "telecomunicaciones"  ~  "Transporte y comunicaciones",
      SectorEconom_O %in% "valores"  ~  "Servicios",
      Ocupacion %in% "Empleado (a)"  ~  "Servicios",
      Ocupacion %in% "Obrero(a)"  ~  "Industria manufacturera",
      Ocupacion %in% "Profesionista empleado"  ~  "Servicios",
      Ocupacion %in% "Comerciante"  ~  "Comercio",
      Ocupacion %in% "Profesionista Independiente"  ~  "Comercio",
      Ocupacion %in% "Oficinista"  ~  "Servicios",
      Ocupacion %in% "Profesionista independiente"  ~  "Comercio",
      Ocupacion_O %in% "EMPLEADO (A)"  ~  "Servicios",
      Ocupacion_O %in% "Chofer"  ~  "Servicios",
      Ocupacion_O %in% "Albañil"  ~  "Construcción",
      Ocupacion_O %in% "Negocio Propio"  ~  "Comercio",
      Ocupacion_O %in% "Taxista"  ~  "Servicios",
      Ocupacion_O %in% "Mecánico"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Maestro"  ~  "Servicios",
      Ocupacion_O %in% "Ayudante general"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Trabaja por su cuenta"  ~  "Comercio",
      Ocupacion_O %in% "Empleada doméstica"  ~  "Servicios",
      Ocupacion_O %in% "Negocio propio"  ~  "Comercio",
      Ocupacion_O %in% "TAXISTA"  ~  "Servicios",
      Ocupacion_O %in% "SOLDADOR"  ~  "Industria manufacturera",
      Ocupacion_O %in% "MECANICO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Maestra"  ~  "Servicios",
      Ocupacion_O %in% "Guardia de seguridad"  ~  "Servicios",
      Ocupacion_O %in% "Guardia"  ~  "Servicios",
      Ocupacion_O %in% "Vigilante"  ~  "Servicios",
      Ocupacion_O %in% "Trilero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Soldador"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Trabajadora doméstica"  ~  "Servicios",
      Ocupacion_O %in% "Operador de tráiler"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "GUARDIA"  ~  "Servicios",
      Ocupacion_O %in% "Jardinero"  ~  "Servicios",
      Ocupacion_O %in% "LIMPIEZA"  ~  "Servicios",
      Ocupacion_O %in% "Operario"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Construcción"  ~  "Construcción",
      Ocupacion_O %in% "EMPLEADA DOMESTICA"  ~  "Servicios",
      Ocupacion_O %in% "Operador"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Policía Municipal"  ~  "Gobierno",
      Ocupacion_O %in% "ALBAÑIL"  ~  "Construcción",
      Ocupacion_O %in% "Cajero (a)"  ~  "Comercio",
      Ocupacion_O %in% "Contratista"  ~  "Comercio",
      Ocupacion_O %in% "MESERO (A)"  ~  "Servicios",
      Ocupacion_O %in% "Ventas"  ~  "Comercio",
      Ocupacion_O %in% "Carpintero"  ~  "Servicios",
      Ocupacion_O %in% "Chofer de Uber"  ~  "Servicios",
      Ocupacion_O %in% "Electricista"  ~  "Servicios",
      Ocupacion_O %in% "Limpieza en casas"  ~  "Servicios",
      Ocupacion_O %in% "OPERADOR"  ~  "Industria manufacturera",
      Ocupacion_O %in% "PINTOR"  ~  "Servicios",
      Ocupacion_O %in% "Limpieza"  ~  "Servicios",
      Ocupacion_O %in% "Tablajero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Técnico"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Vendedor"  ~  "Comercio",
      Ocupacion_O %in% "CARPINTERO"  ~  "Servicios",
      Ocupacion_O %in% "CONTRATISTA"  ~  "Comercio",
      Ocupacion_O %in% "Cocinero"  ~  "Servicios",
      Ocupacion_O %in% "Independiente"  ~  "Comercio",
      Ocupacion_O %in% "Intendente"  ~  "Servicios",
      Ocupacion_O %in% "Recolección de residuos"  ~  "Servicios",
      Ocupacion_O %in% "Supervisor"  ~  "Servicios",
      Ocupacion_O %in% "maestra"  ~  "Servicios",
      Ocupacion_O %in% "ESTILISTA"  ~  "Servicios",
      Ocupacion_O %in% "Eléctrico"  ~  "Servicios",
      Ocupacion_O %in% "Enfermera"  ~  "Servicios",
      Ocupacion_O %in% "Enfermera doméstica"  ~  "Servicios",
      Ocupacion_O %in% "Jornalero"  ~  "Servicios",
      Ocupacion_O %in% "Lava autos"  ~  "Servicios",
      Ocupacion_O %in% "REPARTIDOR"  ~  "Comercio",
      Ocupacion_O %in% "SEGURIDAD"  ~  "Servicios",
      Ocupacion_O %in% "Velador"  ~  "Comercio",
      Ocupacion_O %in% "COCINERA"  ~  "Comercio",
      Ocupacion_O %in% "Campesino"  ~  "Agricultura y ganadería",
      Ocupacion_O %in% "Carnicero"  ~  "Comercio",
      Ocupacion_O %in% "Chofer de tráiler"  ~  "Servicios",
      Ocupacion_O %in% "Cocinera"  ~  "Servicios",
      Ocupacion_O %in% "Dentista"  ~  "Servicios",
      Ocupacion_O %in% "Escolta"  ~  "Servicios",
      Ocupacion_O %in% "GUARDIA DE SEGURIDAD"  ~  "Servicios",
      Ocupacion_O %in% "INTENDENTE"  ~  "Servicios",
      Ocupacion_O %in% "JARDINERO"  ~  "Servicios",
      Ocupacion_O %in% "MAESTRO"  ~  "Servicios",
      Ocupacion_O %in% "Militar"  ~  "Gobierno",
      Ocupacion_O %in% "Músico"  ~  "Servicios",
      Ocupacion_O %in% "NIÑERA"  ~  "Servicios",
      Ocupacion_O %in% "Pintor de casas"  ~  "Servicios",
      Ocupacion_O %in% "Profesor"  ~  "Servicios",
      Ocupacion_O %in% "Reparación de lavadoras"  ~  "Servicios",
      Ocupacion_O %in% "Tornero"  ~  "Servicios",
      Ocupacion_O %in% "Transportista"  ~  "Comercio",
      Ocupacion_O %in% "VENDEDOR (A)"  ~  "Comercio",
      Ocupacion_O %in% "Vendedor ambulante"  ~  "Comercio",
      Ocupacion_O %in% "AYUDANTE GENERAL"  ~  "Servicios",
      Ocupacion_O %in% "Afanador"  ~  "Comercio",
      Ocupacion_O %in% "Ayudante"  ~  "Servicios",
      Ocupacion_O %in% "Bombero"  ~  "Gobierno",
      Ocupacion_O %in% "CHEF"  ~  "Servicios",
      Ocupacion_O %in% "COSTURERA"  ~  "Servicios",
      Ocupacion_O %in% "Chef"  ~  "Servicios",
      Ocupacion_O %in% "Chofer de Didi"  ~  "Servicios",
      Ocupacion_O %in% "Conserje"  ~  "Servicios",
      Ocupacion_O %in% "Doctor (a)"  ~  "Servicios",
      Ocupacion_O %in% "EMPLEADO DE CONFIANZA"  ~  "Servicios",
      Ocupacion_O %in% "EMPRESARIA"  ~  "Comercio",
      Ocupacion_O %in% "EN UN COMERCIO"  ~  "Comercio",
      Ocupacion_O %in% "ESTETICA"  ~  "Servicios",
      Ocupacion_O %in% "Estilista"  ~  "Servicios",
      Ocupacion_O %in% "Gerente"  ~  "Servicios",
      Ocupacion_O %in% "Guardia de Seguridad"  ~  "Servicios",
      Ocupacion_O %in% "INTENDENCIA"  ~  "Servicios",
      Ocupacion_O %in% "Instructor de gimnasio"  ~  "Servicios",
      Ocupacion_O %in% "Jardinería"  ~  "Servicios",
      Ocupacion_O %in% "Lava salas"  ~  "Servicios",
      Ocupacion_O %in% "Lavador de autos"  ~  "Servicios",
      Ocupacion_O %in% "Lavador de camión"  ~  "Servicios",
      Ocupacion_O %in% "Mantenimiento"  ~  "Servicios",
      Ocupacion_O %in% "OPERADOR (A)"  ~  "Servicios",
      Ocupacion_O %in% "OPERADOR TELEFONICO"  ~  "Servicios",
      Ocupacion_O %in% "Operador de Transporte Escolar"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "Operador de autobús"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "Operador de ruta urbana"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "PAQUETERA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "PLOMERO"  ~  "Servicios",
      Ocupacion_O %in% "POLICIA"  ~  "Gobierno",
      Ocupacion_O %in% "POR SU CUENTA"  ~  "Comercio",
      Ocupacion_O %in% "Paquetero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Paquetero de Soriana"  ~  "Comercio",
      Ocupacion_O %in% "Pintor"  ~  "Servicios",
      Ocupacion_O %in% "Plomero"  ~  "Servicios",
      Ocupacion_O %in% "Policía"  ~  "Gobierno",
      Ocupacion_O %in% "Repartidor"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "TABLAJERO"  ~  "Servicios",
      Ocupacion_O %in% "TECNICO"  ~  "Servicios",
      Ocupacion_O %in% "TRABAJA POR SU CUENTA"  ~  "Comercio",
      Ocupacion_O %in% "Tapicero"  ~  "Servicios",
      Ocupacion_O %in% "Tiene negocio en su hogar"  ~  "Servicios",
      Ocupacion_O %in% "Trabaja en un restaurante"  ~  "Servicios",
      Ocupacion_O %in% "VELADOR"  ~  "Comercio",
      Ocupacion_O %in% "VENTAS"  ~  "Comercio",
      Ocupacion_O %in% "gerente"  ~  "Comercio",
      Ocupacion_O %in% "instalador de cantera independiente"  ~  "Industria manufacturera",
      Ocupacion_O %in% "maestro"  ~  "Servicios",
      Ocupacion_O %in% "mecánico"  ~  "Industria manufacturera",
      Ocupacion_O %in% "AGRICULTOR"  ~  "Agricultura y ganadería",
      Ocupacion_O %in% "ALMACENISTA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Abogado"  ~  "Servicios",
      Ocupacion_O %in% "Acomoda en el mercado de abastos"  ~  "Comercio",
      Ocupacion_O %in% "Agricultor"  ~  "Agricultura y ganadería",
      Ocupacion_O %in% "Almacenista"  ~  "Comercio",
      Ocupacion_O %in% "Almacén"  ~  "Comercio",
      Ocupacion_O %in% "Aluminífero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Ama de casa y trabaja"  ~  "Servicios",
      Ocupacion_O %in% "Artesano"  ~  "Comercio",
      Ocupacion_O %in% "Asesor de ventas"  ~  "Comercio",
      Ocupacion_O %in% "Asistente de servicios"  ~  "Servicios",
      Ocupacion_O %in% "Asistente educativo"  ~  "Servicios",
      Ocupacion_O %in% "Asistente medico"  ~  "Servicios",
      Ocupacion_O %in% "Ayudante Taller"  ~  "Servicios",
      Ocupacion_O %in% "Ayudante de Mecánico"  ~  "Servicios",
      Ocupacion_O %in% "Ayudante de mecánico"  ~  "Servicios",
      Ocupacion_O %in% "Ayudante de su papa albañil"  ~  "Servicios",
      Ocupacion_O %in% "Azulejero"  ~  "Servicios",
      Ocupacion_O %in% "BARBERO"  ~  "Servicios",
      Ocupacion_O %in% "Barbero"  ~  "Servicios",
      Ocupacion_O %in% "Barman"  ~  "Servicios",
      Ocupacion_O %in% "Brigadista"  ~  "Servicios",
      Ocupacion_O %in% "COCINERO"  ~  "Servicios",
      Ocupacion_O %in% "Cargador"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Chofer de camión"  ~  "Servicios",
      Ocupacion_O %in% "Chofer de camión urbano"  ~  "Servicios",
      Ocupacion_O %in% "Chofer de transporte de personal"  ~  "Servicios",
      Ocupacion_O %in% "Climas"  ~  "Comercio",
      Ocupacion_O %in% "Club"  ~  "Servicios",
      Ocupacion_O %in% "Coach"  ~  "Servicios",
      Ocupacion_O %in% "Cocinero (a)"  ~  "Servicios",
      Ocupacion_O %in% "Compra fierro metal"  ~  "Comercio",
      Ocupacion_O %in% "Comunicación"  ~  "Servicios",
      Ocupacion_O %in% "Constructora"  ~  "Construcción",
      Ocupacion_O %in% "Criminólogo"  ~  "Servicios",
      Ocupacion_O %in% "DENTISTA"  ~  "Servicios",
      Ocupacion_O %in% "De paquetero"  ~  "Comercio",
      Ocupacion_O %in% "Demostradora"  ~  "Comercio",
      Ocupacion_O %in% "EMPACADOR"  ~  "Comercio",
      Ocupacion_O %in% "EMPLEADO FEDERAL"  ~  "Gobierno",
      Ocupacion_O %in% "EMPLEADO INDEPENDIENTE"  ~  "Comercio",
      Ocupacion_O %in% "EMPLEADO RESTAURANT"  ~  "Servicios",
      Ocupacion_O %in% "EMPLEADO SECTOR CALIDAD"  ~  "Servicios",
      Ocupacion_O %in% "EMPLEADO TORNERO"  ~  "Servicios",
      Ocupacion_O %in% "EMPLEADO UBER"  ~  "Servicios",
      Ocupacion_O %in% "ENFERMERO"  ~  "Servicios",
      Ocupacion_O %in% "Educadora"  ~  "Servicios",
      Ocupacion_O %in% "Ejercicio"  ~  "Servicios",
      Ocupacion_O %in% "Empleada pastelería"  ~  "Servicios",
      Ocupacion_O %in% "Empleado de obra"  ~  "Construcción",
      Ocupacion_O %in% "Empleado de taller de herrería"  ~  "Industria manufacturera",
      Ocupacion_O %in% "En un comedor"  ~  "Servicios",
      Ocupacion_O %in% "Encargada"  ~  "Servicios",
      Ocupacion_O %in% "Encargado"  ~  "Servicios",
      Ocupacion_O %in% "Enfermero"  ~  "Servicios",
      Ocupacion_O %in% "Escolta personal"  ~  "Servicios",
      Ocupacion_O %in% "Estudia y trabaja"  ~  "Servicios",
      Ocupacion_O %in% "FABRICA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "FABRICA DE MADERA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Fotógrafo"  ~  "Servicios",
      Ocupacion_O %in% "GASOLINERA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "GASOLINERO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "GASOLINERO (A)"  ~  "Industria manufacturera",
      Ocupacion_O %in% "GERENTE"  ~  "Comercio",
      Ocupacion_O %in% "GERENTE DE VENTAS"  ~  "Comercio",
      Ocupacion_O %in% "GINEQUITO"  ~  "Servicios",
      Ocupacion_O %in% "GOBIERNO"  ~  "Gobierno",
      Ocupacion_O %in% "Gasolinero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "HERRERO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Herrero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "IMPRENTA"  ~  "Servicios",
      Ocupacion_O %in% "INSTALADOR"  ~  "Industria manufacturera",
      Ocupacion_O %in% "INSTRUCTOR DE GIMNASIO"  ~  "Servicios",
      Ocupacion_O %in% "Ingeniero analista"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Instalador de fibra"  ~  "Industria manufacturera",
      Ocupacion_O %in% "JORNALERO"  ~  "Servicios",
      Ocupacion_O %in% "Jefe Propio"  ~  "Servicios",
      Ocupacion_O %in% "Jefe almacén"  ~  "Servicios",
      Ocupacion_O %in% "Jefe de meseros"  ~  "Servicios",
      Ocupacion_O %in% "Jefe de personal"  ~  "Servicios",
      Ocupacion_O %in% "LLANTERO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Limpiavidrios"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Limpieza de solares o jardines"  ~  "Servicios",
      Ocupacion_O %in% "MAESTRA"  ~  "Servicios",
      Ocupacion_O %in% "MAESTRIA"  ~  "Servicios",
      Ocupacion_O %in% "MANTENIMIENTO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "MANTENIMIENTO FABRICA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "MONTACARGUISTA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Maestra particular"  ~  "Servicios",
      Ocupacion_O %in% "Mantenimiento a residencias"  ~  "Servicios",
      Ocupacion_O %in% "Mecánico Electricista"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Medicina"  ~  "Servicios",
      Ocupacion_O %in% "Mesera"  ~  "Servicios",
      Ocupacion_O %in% "Musico"  ~  "Servicios",
      Ocupacion_O %in% "Negocio"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Negocio camiones volteo"  ~  "Industria manufacturera",
      Ocupacion_O %in% "OBRERO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "OPERADOR DE RUTA URBANA"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "OPERADORA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "OPERARIA"  ~  "Industria manufacturera",
      Ocupacion_O %in% "OPERARIO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Oficinista"  ~  "Servicios",
      Ocupacion_O %in% "Operador de camión"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Operador de maquinaria"  ~  "Industria manufacturera",
      Ocupacion_O %in% "Operaria"  ~  "Industria manufacturera",
      Ocupacion_O %in% "PANADERO"  ~  "Comercio",
      Ocupacion_O %in% "PAQUETER SMART"  ~  "Industria manufacturera",
      Ocupacion_O %in% "PAQUETERIA SORIAN"  ~  "Industria manufacturera",
      Ocupacion_O %in% "PASTELERIA"  ~  "Comercio",
      Ocupacion_O %in% "PEPENADOR"  ~  "Comercio",
      Ocupacion_O %in% "PINTURA"  ~  "Servicios",
      Ocupacion_O %in% "PROMOTOR"  ~  "Comercio",
      Ocupacion_O %in% "PROMOTORA"  ~  "Comercio",
      Ocupacion_O %in% "PUESTO"  ~  "Comercio",
      Ocupacion_O %in% "Panadero"  ~  "Comercio",
      Ocupacion_O %in% "Paquetera"  ~  "Comercio",
      Ocupacion_O %in% "Paquetería voluntaria de supermercado"  ~  "Comercio",
      Ocupacion_O %in% "Parrillero"  ~  "Comercio",
      Ocupacion_O %in% "Pepenador de Pet"  ~  "Comercio",
      Ocupacion_O %in% "Peón de albañil"  ~  "Servicios",
      Ocupacion_O %in% "Por cuenta propia"  ~  "Servicios",
      Ocupacion_O %in% "Prefecta"  ~  "Servicios",
      Ocupacion_O %in% "Profesora"  ~  "Servicios",
      Ocupacion_O %in% "RECEPCIONISTA"  ~  "Servicios",
      Ocupacion_O %in% "RECICLADORA"  ~  "Comercio",
      Ocupacion_O %in% "REPARADOR DE CANCELES"  ~  "Comercio",
      Ocupacion_O %in% "REPARTIDOR EN MOTO"  ~  "Comercio",
      Ocupacion_O %in% "Realiza varios trabajos por su cuenta para la casa"  ~  "Comercio",
      Ocupacion_O %in% "Recolector de Pet"  ~  "Servicios",
      Ocupacion_O %in% "Recolector de aluminio"  ~  "Comercio",
      Ocupacion_O %in% "Repartidor de gas"  ~  "Servicios",
      Ocupacion_O %in% "Repostera"  ~  "Servicios",
      Ocupacion_O %in% "Ritualista"  ~  "Servicios",
      Ocupacion_O %in% "SECRETARIA"  ~  "Servicios",
      Ocupacion_O %in% "SEGURIDAD PUBLICA"  ~  "Gobierno",
      Ocupacion_O %in% "SERVICIO PUBLICO"  ~  "Gobierno",
      Ocupacion_O %in% "SERVICIOS PRIMARIOS"  ~  "Industria manufacturera",
      Ocupacion_O %in% "SUPERVISOR"  ~  "Construcción",
      Ocupacion_O %in% "Se dedica a la construcción"  ~  "Construcción",
      Ocupacion_O %in% "Servicio de limpieza"  ~  "Servicios",
      Ocupacion_O %in% "Servidor público"  ~  "Gobierno",
      Ocupacion_O %in% "Servidumbre"  ~  "Servicios",
      Ocupacion_O %in% "TALLER"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TAQUERO"  ~  "Servicios",
      Ocupacion_O %in% "TEC MANTENIMIENTO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TECNICO EN AUDIO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TECNICO EN MANTENIMIENTO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TECNICO EN REFRIGERACION"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TECNICO MWECANICO"  ~  "Servicios",
      Ocupacion_O %in% "TORNERO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TRABAJO"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TRABAJO DOMESTICO"  ~  "Servicios",
      Ocupacion_O %in% "TRABAJO EN HOTEL"  ~  "Servicios",
      Ocupacion_O %in% "TRABAJOS EVENTUALES"  ~  "Industria manufacturera",
      Ocupacion_O %in% "TRAILERO"  ~  "Transporte y comunicaciones",
      Ocupacion_O %in% "TRANSPORTISTA-FLETES"  ~  "Comercio",
      Ocupacion_O %in% "Talachero"  ~  "Comercio",
      Ocupacion_O %in% "Taller torno"  ~  "Comercio",
      Ocupacion_O %in% "Taquero"  ~  "Comercio",
      Ocupacion_O %in% "Tienda autoservicio"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja con su Tío"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja con su esposo"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja en casino"  ~  "Servicios",
      Ocupacion_O %in% "Trabaja en su casa"  ~  "Servicios",
      Ocupacion_O %in% "Trabaja en su propio transporte"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja en un rancho"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja en una tienda de abarrotes"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja servicio de limpieza"  ~  "Servicios",
      Ocupacion_O %in% "Trabaja tienda"  ~  "Comercio",
      Ocupacion_O %in% "Trabaja y Estudia"  ~  "Servicios",
      Ocupacion_O %in% "Trabaja y estudia"  ~  "Servicios",
      Ocupacion_O %in% "Trabajadora social"  ~  "Servicios",
      Ocupacion_O %in% "Tránsito"  ~  "Comercio",
      Ocupacion_O %in% "Técnico  en refrigeración"  ~  "Servicios",
      Ocupacion_O %in% "Técnico en Refrigeración"  ~  "Servicios",
      Ocupacion_O %in% "Técnico en climas"  ~  "Servicios",
      Ocupacion_O %in% "Técnico en mantenimiento"  ~  "Servicios",
      Ocupacion_O %in% "Técnico en sistemas"  ~  "Servicios",
      Ocupacion_O %in% "Uber eat"  ~  "Servicios",
      Ocupacion_O %in% "VENDEDOR"  ~  "Comercio",
      Ocupacion_O %in% "VIGILANTE"  ~  "Servicios",
      Ocupacion_O %in% "Venta de material para reciclar"  ~  "Comercio",
      Ocupacion_O %in% "Yesero"  ~  "Industria manufacturera",
      Ocupacion_O %in% "barbero"  ~  "Servicios",
      Ocupacion_O %in% "comerciante"  ~  "Comercio",
      Ocupacion_O %in% "educación"  ~  "Servicios",
      Ocupacion_O %in% "empleo eventual"  ~  "Comercio",
      Ocupacion_O %in% "estudia y trabaja"  ~  "Servicios",
      Ocupacion_O %in% "independiente"  ~  "Comercio",
      Ocupacion_O %in% "jardinero"  ~  "Servicios",
      Ocupacion_O %in% "jornalero"  ~  "Comercio",
      Ocupacion_O %in% "mozo"  ~  "Comercio",
      Ocupacion_O %in% "operario de camión de pipa de agua"  ~  "Comercio",
      Ocupacion_O %in% "pintor"  ~  "Comercio",
      Ocupacion_O %in% "pulidor de.pisos"  ~  "Servicios",
      Ocupacion_O %in% "reclutamiento"  ~  "Comercio",
      Ocupacion_O %in% "recolecta papel y botellas"  ~  "Servicios",
      Ocupacion_O %in% "salud"  ~  "Servicios",
      Ocupacion_O %in% "trabaja en trailes trasportando"  ~  "Industria manufacturera",
      Ocupacion_O %in% "trabaja y estudia"  ~  "Comercio",
      TRUE ~ "Otro"
    ))





# Testing random forest for informal/formal model -------------------------

set.seed(123)

## Evaluación del modelo
# Create the training and test sets using the createDataPartition function
train_prop <- 0.7
train_index <- createDataPartition(enoe_b$informal, p = train_prop, list = FALSE)

train_set <- enoe_b[train_index, ]
test_set <- enoe_b[-train_index, ]

bag.boston = randomForest(informal~., data = train_set, ntree = 500, importance = TRUE)

yhat.bag = predict(bag.boston, newdata= test_set)

aaa <- as_tibble(yhat.bag)

confusionMatrix(test_set$informal, aaa$value)
## Accuracy del 75.45%



## Nested CV
# Load required libraries
library(caret)
library(randomForest)
set.seed(123)

# Setup 5-fold cross validation for outer loop
cv_outer <- createFolds(enoe_b$informal, k = 5, list = TRUE, returnTrain = TRUE)

# Empty vector to store results
accuracy_outer <- vector()

for(i in 1:5) {
  
  # Training and testing datasets for outer loop
  train_outer <- enoe_b[cv_outer[[i]], ]
  test_outer  <- enoe_b[-cv_outer[[i]], ]
  
  # Inner loop: hyperparameter tuning using caret's train function with 3-fold cross validation
  ctrl <- trainControl(method = "cv", number = 3)
  tuneGrid <- expand.grid(.mtry = seq(from = 2, to = 10, by = 2)) #example grid, change as per your requirement
  
  # Train the model
  model <- train(informal ~ ., data = train_outer, method = "rf", trControl = ctrl, tuneGrid = tuneGrid, ntree = 100)
  
  # Predict on outer test set
  pred_outer <- predict(model, newdata = test_outer)
  
  # Confusion matrix
  CM <- confusionMatrix(test_outer$informal, pred_outer)
  
  # Get accuracy and store in results vector
  accuracy_outer[i] <- CM$overall['Accuracy']
  print(i)
}

# Output average accuracy
mean(accuracy_outer)

## 80.4%


# Testing SVM for informal/formal model -----------------------------------

library(e1071)


set.seed(123)
train_prop <- 0.7
train_index <- createDataPartition(enoe_b$informal, p = train_prop, list = FALSE)

train_data <- enoe_b[train_index, ]
test_data <- enoe_b[-train_index, ]

train_data <- train_data  %>% 
  select(!c(factor)) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer))



test_data <- test_data %>% 
  select(!c(factor)) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer))



# Create an SVM model
svm_model <- svm(informal ~ ., data = train_data, kernel = "radial")

# Predict the labels for the test data
predicted_labels <- predict(svm_model, newdata = test_data)

predicted_labels <- as_tibble(predicted_labels) %>% 
  mutate(value = if_else(value < 1.5, 1, 2))

# Calculate the accuracy of the predicted labels compared to the true test labels
accuracy <- sum(predicted_labels$value == test_data$informal) / length(test_data$informal)

# Print the accuracy
cat("Accuracy:", accuracy)


# tiene un accuracy del 0.78


### Nested cross validation
# Load required libraries
library(e1071)
library(caret)

set.seed(123)

# Set up 5-fold cross validation for outer loop
cv_outer <- createFolds(enoe_b$informal, k = 5, list = TRUE, returnTrain = TRUE)

# Empty vector to store results
accuracy_outer <- vector()

for (i in 1:5) {
  
  # Training and testing datasets for outer loop
  train_outer <- enoe_b[cv_outer[[i]], ]
  test_outer <- enoe_b[-cv_outer[[i]], ]
  
  # Inner loop: hyperparameter tuning using caret's train function with 3-fold cross-validation
  ctrl <- trainControl(method = "cv", number = 3)
  tuneGrid <- expand.grid(.sigma = c(0.5, 1), .C = c(1, 2)) # example grid, change as per your requirement
  
  # Train the model
  model <- train(informal ~ ., 
                 data = train_outer, 
                 method = "svmRadial", 
                 preProcess = c("center", "scale"), 
                 trControl = ctrl, 
                 tuneGrid = tuneGrid)
  
  # Predict on outer test set
  pred_outer <- predict(model, newdata = test_outer)
  
  # Confusion matrix
  CM <- confusionMatrix(test_outer$informal, pred_outer)
  
  # Get accuracy and store in results vector
  accuracy_outer[i] <- CM$overall['Accuracy']
}

# Output average accuracy
mean(accuracy_outer)



# Testing Logit Model for informal/formal model ----------------------------
set.seed(123)

## Evaluación del modelo
# Create the training and test sets using the createDataPartition function
train_prop <- 0.7
train_index <- createDataPartition(enoe_b$informal, p = train_prop, list = FALSE)

train_set <- enoe_b[train_index, ]
test_set <- enoe_b[-train_index, ]

train_data <- train_set  %>% 
  select(informal, genero, ocupacion, sector, edad, escolaridad, municipio) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer)) 



test_data <- test_set %>% 
  select(informal, genero, ocupacion, sector, edad, escolaridad, municipio) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer))


train_data$informal
modelo <- glm(informal ~ ., data = train_data, family = "binomial")

predicciones <- predict(modelo, newdata = test_data)

aaa <- as_tibble(predicciones) %>% 
  mutate(value = scale(value),
         valor = as.factor(case_when(
           value > median(value) ~ 1,
           TRUE ~ 0
         )))


confusionMatrix(test_data$informal, aaa$valor)


#69.52


## con nested cv
library(caret)

# Convert all predictor variables to factors
data <- enoe_b %>%
  select(informal, genero, ocupacion, sector, edad, escolaridad, municipio) %>%
  mutate(across(genero:municipio, as.factor)) %>%
  mutate(across(genero:municipio, as.integer))

# Create 5-fold CV for the outer loop
folds <- createFolds(data$informal, k = 5, list = TRUE, returnTrain = TRUE)

accuracy <- vector() # Store accuracy for each outer fold

for(i in 1:5){
  # Create training and testing datasets
  train_data <- data[folds[[i]], ]
  test_data  <- data[-folds[[i]], ]
  
  # Control grid for the inner loop
  control <- trainControl(method="cv", number=3) 
  
  # Model fitting in the inner loop
  model <- train(informal ~ ., data=train_data, method="glm", trControl=control, family="binomial")
  
  # Predict and calculate accuracy on the test data
  predictions <- predict(model, newdata = test_data)
  cm <- confusionMatrix(predictions, test_data$informal)
  accuracy[i] <- cm$overall['Accuracy']
}

# Mean accuracy over all outer folds
mean_accuracy <- mean(accuracy)
print(paste("The mean accuracy over all folds was", round(mean_accuracy, 3)))






# Testing KNN for informal/formal model -----------------------------------

## Cargo la libreria
library(class)
library(caret)

# Preparación del test y train set
set.seed(123)
train_prop <- 0.7
train_index <- createDataPartition(enoe_b$informal, p = train_prop, list = FALSE)


train_set <- enoe_b[train_index, ]
test_set <- enoe_b[-train_index, ]

train_knn <- train_set  %>% 
  select(!c(sector,factor)) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer)) %>% 
  mutate(across(genero:municipio, scale)) %>% 
  select(!informal)


test_knn <- test_set %>% 
  select(!c(sector,factor)) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer)) %>% 
  mutate(across(genero:municipio, scale)) %>% 
  select(!informal)


test_pred <- knn(
  train = train_knn, 
  test = test_knn,
  cl = train_set$informal, 
  k = 10
)



# Step 5: Evaluate the model
# Assuming your test labels are stored in test_labels in test_data

# Calculate the accuracy of the predicted labels compared to the true test labels
accuracy <- sum(test_pred == test_set$informal) / length(test_set$informal)



# Print the accuracy
cat("Accuracy:", accuracy)

## El KNN muestra un accuracy de 0.772

## nested cv
# Load necessary library
library(class)
library(caret)

# Preprocessing
data_knn <- enoe_b  %>% 
  select(!c(sector,factor)) %>% 
  relocate(informal, .before = everything()) %>% 
  mutate(across(genero:municipio, as.factor))%>% 
  mutate(across(genero:municipio, as.integer)) %>% 
  mutate(across(genero:municipio, scale))

# Create 5-fold CV for the outer loop
folds <- createFolds(data_knn$informal, k = 5, list = TRUE, returnTrain = TRUE)

accuracy <- vector() # Store accuracy for each outer fold

for(i in 1:5){
  # Create training and testing datasets
  train_data <- data_knn[folds[[i]], ]
  test_data  <- data_knn[-folds[[i]], ]
  
  # Control grid for the inner loop
  control <- trainControl(method="cv", number=3) 
  
  # Grid for the k parameter in KNN 
  grid <- expand.grid(k = seq(from = 1, to = 10, by = 1)) 
  
  # Model fitting in the inner loop
  model <- train(informal ~ ., data=train_data, method="knn", tuneGrid = grid, trControl=control)
  
  # Predict and calculate accuracy on the test data
  predictions <- predict(model, newdata = test_data)
  cm <- confusionMatrix(predictions, test_data$informal)
  accuracy[i] <- cm$overall['Accuracy']
  
}

# Mean accuracy over all outer folds
mean_accuracy <- mean(accuracy)
print(paste("The mean accuracy over all folds was", round(mean_accuracy, 3)))




# Creation of formal/informal base with RF --------------------------------

set.seed(123)


enoe_bsf <-enoe_b %>% 
  mutate(sector = as.factor(sector),
         ocupacion = as.factor(ocupacion),
         genero = as.factor(genero),
         edad = as.factor(edad),
         escolaridad = as.factor(escolaridad),
         municipio = as.factor(municipio)
  ) %>% 
  select(!factor)

bag.boston = randomForest(informal~ . , data = enoe_bsf, ntree = 40, importance = TRUE)

importance(bag.boston)


od_bien <- od %>% 
  filter(!edad %in% c("6-7", "8-11"),
         !escolaridad %in% "Otro") %>% 
  transmute(sector = as.factor(sector),
            ocupacion = as.factor(ocupacion),
            genero = as.factor(genero),
            edad = as.factor(edad),
            escolaridad = as.factor(escolaridad),
            municipio = as.factor(municipio),
            `H-P-V`
  )


mean(is.na(od_bien))
mean(is.na(od_bien$sector))
mean(is.na(od_bien$ocupacion))
mean(is.na(od_bien$genero))
mean(is.na(od_bien$edad))
mean(is.na(od_bien$escolaridad))
mean(is.na(od_bien$municipio))


## Verify levels of factos
levels(od_bien$sector)
levels(enoe_bsf$sector)


levels(od_bien$ocupacion)
levels(enoe_bsf$ocupacion)

levels(od_bien$genero)
levels(enoe_bsf$genero)

levels(od_bien$edad)
levels(enoe_bsf$edad)

levels(od_bien$escolaridad)
levels(enoe_bsf$escolaridad)

levels(od_bien$municipio)
levels(enoe_bsf$municipio)




yhat.bag = predict(bag.boston, newdata= od_bien)





aaa <- as_tibble(yhat.bag)

mean(aaa$value == 1)

nuevo_censo <- cbind(od_bien, aaa)

nuevo_censo <- nuevo_censo %>% 
  rename(informal = value)




## All the previous steps were for building the new database with the predicted
## classifications of workers as either formal or informal. 
## now, a brief analysis of the distrivution is going to be done



# Distributions of informals in  enoe v.  predicted OD --------------------
## Creation of the OD survey with factors, joining by identifier HPV
eodh_base <- read_csv("base_eodh/datos_limpios_tiempos.csv") %>% 
  select(c("H-P-V", "FE"))

eodh <- nuevo_censo %>% 
  left_join(eodh_base, by = "H-P-V", suffix = c("",".y")) %>% 
  select(!ends_with(".x"))


## open ENOE for real distributions
enoe <- read_csv("bases_creadas/enoe_informalidad_v281023.csv")


## apply survey design
svy_eodh <- as_survey_design(eodh, weights = FE)
svy_enoe <- as_survey_design(enoe, weights = factor)


## observe proportion of informals within each sector

# Predicted data
svy_eodh %>% 
  group_by(sector, informal) %>% 
  summarise(prom= survey_mean(vartype = NULL),
            survey_total(vartype = NULL)
  ) %>% 
  filter(informal == 1)

## Real data (according to ENOE)
svy_enoe %>% 
  group_by(sector, informal) %>% 
  summarise(survey_mean(vartype = NULL)) %>% 
  filter(informal == 1)

## Observe distribution of informals among the different sectors
# Predicted data
svy_eodh %>% 
  group_by(informal, sector) %>% 
  summarise(prom= survey_mean(vartype = NULL),
            survey_total(vartype = NULL)
  ) %>% 
  filter(informal == 1)

## Real data (according to ENOE)
svy_enoe %>% 
  group_by(informal, sector) %>% 
  summarise(survey_mean(vartype = NULL)) %>% 
  filter(informal == 1)



## Conclusion: there is a need for a second model that reclassifies the large amount of "Other"
## in the OD survey



# Random forest for classifying construction/commerce ---------------------


eodh <- nuevo_censo
enoe <- read_csv("bases_creadas/enoe_informalidad_v281023.csv") %>% 
  filter(sector %in% c("Comercio", "Construcción"))


## Separo las observaciones que hace falta ubicar en alguna categoría. 
por_ubicar <- eodh %>% 
  filter(sector %in% "Otro") %>% 
  mutate(across(c(genero, ocupacion, sector,edad, escolaridad, informal, municipio ), as.factor))

set.seed(123)

enoe_rf <- enoe %>% 
  select(!c(factor, ocupacion, informal)) %>% 
  filter(!escolaridad %in% "Otro") %>% 
  mutate(across(genero:municipio, as.factor)) 


bag.boston = randomForest(sector~., data = enoe_rf, ntree = 25, importance = TRUE)


importance(bag.boston)

yhat.bag = predict(bag.boston, newdata= por_ubicar)


aaa <- as_tibble(yhat.bag)



nuevo_eodh <- cbind(por_ubicar, aaa)

eodh_nueva <- nuevo_eodh %>% 
  rename(sector_nuevo = value)

eodh_todo <- eodh %>% 
  left_join(select(eodh_nueva, "H-P-V", sector_nuevo), by = "H-P-V", ) %>% 
  mutate(sector = case_when(
    !sector %in% "Otro" ~ sector,
    sector %in% "Otro" ~ sector_nuevo
  ))

od <- read_csv("base_eodh/datos_limpios_tiempos.csv")

eodh_final <- eodh_todo %>% 
  left_join(od, by = "H-P-V", suffix = c("", ".y")) %>% 
  select(!ends_with(".y"))












# Distribution of new classifications -------------------------------------



eodh <- eodh_final

## open ENOE for real distributions
enoe <- read_csv("bases_creadas/enoe_informalidad_v281023.csv")


## apply survey design
svy_eodh <- as_survey_design(eodh, weights = FE)
svy_enoe <- as_survey_design(enoe, weights = factor)


## observe proportion of informals within each sector

# Predicted data
svy_eodh %>% 
  group_by(sector, informal) %>% 
  summarise(prom= survey_mean(vartype = NULL),
            survey_total(vartype = NULL)
  ) %>% 
  filter(informal == 1)

## Real data (according to ENOE)
svy_enoe %>% 
  group_by(sector, informal) %>% 
  summarise(survey_mean(vartype = NULL)) %>% 
  filter(informal == 1)

## Observe distribution of informals among the different sectors
# Predicted data
svy_eodh %>% 
  group_by(informal, sector) %>% 
  summarise(prom= survey_mean(vartype = NULL),
            survey_total(vartype = NULL)
  ) %>% 
  filter(informal == 1)

## Real data (according to ENOE)
svy_enoe %>% 
  group_by(informal, sector) %>% 
  summarise(survey_mean(vartype = NULL)) %>% 
  filter(informal == 1)
