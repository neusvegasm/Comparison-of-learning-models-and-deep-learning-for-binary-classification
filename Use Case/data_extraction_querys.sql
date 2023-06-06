/* PERSONES */
SELECT codi_persona     id_persona,
       sexe_app         sexe,
       trunc(dnaixement) dnaixement,
       clocal,
       cpostal
  FROM persones p
  
  
/* LOCALITATS */
SELECT codi clocal, nucli, comarca, ulocal from localitats

/* COMARQUES*/ 
select codi comarca, descripcio, provincia from comarques

/* PROVIN */
select codi provincia, provincia descripcio from provin  

/* UNIONS_LOCALS */
select codi ulocal, descripcio, provincia, uci from unions_locals

/* UNIONS_COMAR_INTER */
select codi uci, descripcio, abrevia from unions_comar_inter
  

/* AFILIATS */
SELECT (SELECT p.codi_persona
          FROM persones p
         WHERE dni = a.dni)    id_persona,
         codictreb,
         profesio,
         categoria,
         contracte_afil,
         slaboral,
         categ2011 categ, 
         origen,
         tipaf,
         tippag,
         tippag_periodo,         
         trunc(dalta) dalta,
         trunc(dbaixa) dbaixa,
         ulocal
  FROM afiliats a 
  
  
  
/* TIPUS_CONTRACTE */
select codi contracte_afil, descripcio from BAT.tipus_contracte

/* SITUACIO LABORAL */
select codi slaboral,descripcio from situacio_laboral

/* CATEG_AFIL*/
  select codi categ, descripcio from bat.categ_afil
  

/* ORIGEN_AFILIACIO */
select codi origen, descripcio from BAT.ORIGEN_AFILIACIO  

/* TIPUS_AFILILACIO*/
select codi tipaf,descripcio, TQ_CAT from tipus_afiliacio

/* TIPUS_PAGAMENT */
select codi tippag, descripcio from tipus_pagament


/* BAIXA */
SELECT (SELECT codi_persona
          FROM persones
         WHERE dni = baixa.dni)    id_persona,
       codictreb,
       profesio,
       categoria,
       tipaf,
       tippag,
       dalta,
       dbaixa,
       causa,
       causa2,
       ulocal,
       contracte_Afil,
       slaboral,
       categ2011                   categ
  FROM baixa
  
/* CBAFILIATS */
select codi causa,descripcio from cbafiliats

/* CBAFILIATS2 */
select codi_motiu1 causa, codi_motiu2 causa2, descripcio from cbafiliats2
  


/* COTITZACIONS */
select (select codi_persona from persones where dni=c.dni) id_persona, fac, pag, tippag, any_coti, mes, tipaf, trunc(datapag) datapag from cotitzacions c




/* EMPRESES */
select codictreb,
       nomempresa,
       nomemp2,
       nomcomercial,
       numss,
       nif,
       clocal,
       ram,
       sector,
       ntreball,
       conveni,
       conveni2,
       cnae,
       trunc(dalta) dalta,
       trunc(dbaixa) dbaixa,
       nomina      
 from empreses
 
 
 /* RAMS */
 select ram,descripcio from rams
 
 select ram, codi sector, descripcio, codi_agrupacio, agrupacio from sectors
 
 /* convenis */
 select conveni,descripcio from convenis
 
 /* cnaes */
 select cnae,descripcio from cnaes
 
 
 /* PREAVIS */
 select numpreavis,codictreb, tipelec, promotor from preavis

/* ACTA */
select numacta,numpreavis,codictreb,dvota, numss, cif nif, trebcompu  from acta where califiacta<>'N'

/* COLEGI */
select numacta,colegi, elech,elecd, repaelegir, voth, votd, votcump, votblanc,votnul, repelegits from colegi  where numacta in (select numacta from acta where califiacta<>'N')

/* CANDTURA */
select numacta, colegi, sigla, presen, elegit, nvots From candtura where numacta in (select numacta from acta where califiacta<>'N')

/* CANDIDAT */
select numacta,colegi, sigla, numord, (select codi_persona from persones where dni=candidat.dni) id_persona, dalta, dbaixa   From candidat where numacta in (select numacta from acta where califiacta<>'N')

/* AGCTEESS*/
select codi codi_agrupacio, nom, ctrebprin from agcteess where codi like 'GES%'

/* RELAGREESS */
select codi codi_agrupacio, codictreb from relagreess where codi like 'GES%'

/* SIGLA */
select codi sigla, descripcio from siglas



/* FORMACIO */
select  (Select codi_persona from persones where dni=i.nif) id_persona,
                 a.accionnombre                                           curs,
                g.fechainicio                                            inici,
                g.FECHAFIN                                               fi,
                a.HPRESENCIALES + a.HTELEFORMACION + a.HDISTANCIA        hores,
                NVL (i.finaliza, 'S')                                    finalitza,
                (select  expedientenombre from nougesform.expedientes where expedienteid=a.expedienteid) nom_expediente,
                (select tipo from nougesform.prog where progid=(select e.progid from nougesform.expedientes e where expedienteid=a.expedienteid)) tipo
FROM grupos g, inscripciones i, acciones a
WHERE g.grupoid = i.grupoid AND i.inicia = 'S' AND g.accionid = a.accionid and i.nif in (select dni from persones where dni=i.nif)


/* ASSESSORAMENT */
/* Formatted on 17/04/2023 17:53:37 (QP5 v5.391) */
SELECT (SELECT codi_persona
          FROM persones
         WHERE dni = va.dni)            id_persona,
       v.dalta,
       v.clocal,
       (SELECT codi_motiu
          FROM asp.asp_visita_resultat
         WHERE codi_visita = v.codi)    codi_motiu,
       (SELECT codi_submotiu
          FROM asp.asp_visita_resultat
         WHERE codi_visita = v.codi)    codi_submotiu,
       (SELECT codi_resultat
          FROM asp.asp_visita_resultat
         WHERE codi_visita = v.codi)    codi_resultat,
        (case 
                    when ASP.RET_ISRESPONSABLE_GJ_DATA((select dni from asp.asp_Asesores a where a.codi=v.codi_asesor), v.dalta)='S' then 'JURIDIC' 
                    else 'SINDICAL' 
                 end) TIPUS
  FROM asp.asp_visitas v, asp.asp_visitas_asistentes va
 WHERE v.codi = va.codi_visita AND v.realizada = 'S'
 
 


/* MOTIUS */
select codi codi_motiu, descripcio from asp.ASP_VISITA_RES_TIPO

/* SUBMOTIUS */
select codi_motiu,codi_submotiu, descripcio from asp.ASP_VISITA_RES_SUBTIPO

/* RESULTATS VISITA */
select codi codi_resultat, descripcio from asp.ASP_VISITA_RES_MOTIU

/* SERVEIS_GTJ */
select * from (
select (select codi_persona from persones where dni=cl.per_dni) id_persona, ret_materia_incial(cc.exp_codi_intern,cc.carp_codi) minicial, c.data_alta from clients_carpetes cc, clients cl, carpetes c where c.exp_Codi_intern=cc.exp_Codi_intern and c.codi=cc.carp_codi and cl.codi=cc.clie_cli_codi
) where id_persona is not null 



