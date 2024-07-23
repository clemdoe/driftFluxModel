MODULE EosIAPWSRead
!
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NRC Prime Contract no. NRC-04-97-046 (Task #2)
!
!  Purpose: This module contains routines to read light water steam tables from
!           data file (i.e. problemName.h2o, trch2o, or tpfh2onew) or load the
!           steam tables from an ascii file.
!
!  Programmed by Jay Spore, ISL, Date (07/05)
!
  USE IntrType, ONLY: sdk, sik
!
  IMPLICIT NONE
!
CONTAINS
!######################################################
      SUBROUTINE newstREAD(n, nuse, fid_lcl)
!
!  newstread reads and initializes steam tables from new
!  thermodynamic properties file data for light water
!
!  Cognizant engineer: rwt.
!
         USE Io
         USE EosIAPWSData, ONLY: tables, AllocIAPWSTables, newrecord
         USE EosIAPWSData, ONLY: ptrip, vtrip, ttrip, tcrit, pcrit, vcrit, tmin, pmin, tmax, pmax
         USE EosIAPWSData, ONLY: ntemp, npres, nsubcrittemp, nsupcrittemp, nsattemp
         USE EosIAPWSData, ONLY: ofirstsattemp, olastsattemp, otemptrip1, otempcrit1, nsubcritpres, nsupcritpres, nsatpres
         USE EosIAPWSData, ONLY: ofirstsatpres, olastsatpres, oprestrip2, oprescrit2, ofirstsupcritpres, olastsupcritpres
         USE EosIAPWSData, ONLY: ptable1, ltable1, ntable1, stable1, ptable2, ltable2, ntable2, stable2
         USE EosIAPWSData, ONLY: ptable3, ltable3, ntable3, stable3, ptable4, ltable4, ntable4, stable4
         USE EosIAPWSData, ONLY: ptable5, ltable5, ntable5, stable5, ptable6, ltable6, ntable6, stable6
         USE EosIAPWSData, ONLY: ptable7, ltable7, ntable7, stable7, ptable8, ltable8, ntable8, stable8
         USE EosIAPWSData, ONLY: ptable9, ltable9, ntable9, stable9, ptable10, ltable10, ntable10, stable10
!
         IMPLICIT NONE
!
         INTEGER(sik), INTENT(IN) :: n, fid_lcl
         INTEGER(sik), INTENT(OUT) :: nuse
         INTEGER(sik) :: i, ios, ntot
!
         CHARACTER(LEN=80), DIMENSION(2) :: record
!
!--rewind thermodynamic properties file
!
         REWIND n
!
!--get thermodynamic properties file title, and information about the
!--generating program
!
         READ (n, end=10, ERR=20, IOSTAT=ios) record(1)
         READ (n, end=10, ERR=20, IOSTAT=ios) record(2)
         newrecord = record
         DO i = 1, SIZE(outunt)
!            WRITE (outunt(i), '(a)') ' Reading IAPWS H2O table from the file with the following header:'
!            WRITE (outunt(i), '(1x, a/1x, a)') record(1:2)
            WRITE (fid_lcl, '(a)') ' Reading IAPWS H2O table from the file with the following header:'
            WRITE (fid_lcl, '(1x, a/1x, a)') record(1:2)
         END DO
!
!--get triple point and critical point data, minimum and maximum
!--temperatures and pressures, table statistics, and table pointers
!
         READ (n, end=10, ERR=20, IOSTAT=ios) ttrip, ptrip, vtrip, tcrit, pcrit, vcrit, tmin, tmax, pmin, &
        & pmax
!
         READ (n, end=10, ERR=20, IOSTAT=ios) ntemp, npres, nsubcrittemp, nsupcrittemp, nsattemp, &
        & ofirstsattemp, olastsattemp, otemptrip1, otempcrit1, nsubcritpres, nsupcritpres, nsatpres, &
        & ofirstsatpres, olastsatpres, oprestrip2, oprescrit2, ofirstsupcritpres, olastsupcritpres
!
         READ (n, end=10, ERR=20, IOSTAT=ios) ptable1, ltable1, ntable1, stable1, ptable2, ltable2, ntable2, &
        & stable2, ptable3, ltable3, ntable3, stable3, ptable4, ltable4, ntable4, stable4, ptable5, ltable5, &
        & ntable5, stable5, ptable6, ltable6, ntable6, stable6, ptable7, ltable7, ntable7, stable7, ptable8, &
        & ltable8, ntable8, stable8, ptable9, ltable9, ntable9, stable9, ptable10, ltable10, ntable10, &
        & stable10
!
!--get number of words in steam tables
!
         READ (n, end=10, ERR=20, IOSTAT=ios) ntot
!
!--check number of words in steam tables against number of words
!--available for steam tables storage
!
         CALL AllocIAPWSTables(ntot, nuse)
!
         IF (nuse ==-1) GO TO 50
!
!rex+ Feb 2001
         WRITE (iout, 1101) nuse, ntot
1101     FORMAT (' IAPWS Steam Table newstread: nuse=', i8, ', ntot=', i8)
!rex-

!
!--get steam tables
!
         READ (n, end=10, ERR=20, IOSTAT=ios) (tables(i), i=1, ntot)
         GO TO 50
!
!--premature end of data encountered
!
10       CONTINUE
         DO i = 1, SIZE(outunt)
            WRITE (outunt(i), 1001)
         END DO
         GO TO 40
!
!--error reading steam table data
!
20       CONTINUE
         DO i = 1, SIZE(outunt)
            WRITE (outunt(i), 1002) ios
         END DO
         GO TO 40
!
40       nuse = - 1
!
!--done
!
50       RETURN
!
!
1001     FORMAT ('0***** end of data encountered reading thermodynamic property file')
1002     FORMAT ('0***** read error encountered reading thermodynamic property file, iostat =', i4)
!
!
      END SUBROUTINE newstREAD
!
!##############################################################################
!
      SUBROUTINE GetSteamTables(tbl_opt,fid_msg)
!
!         USE BadInput
!         USE EosData, ONLY: nActEos, actFluids, eosH2o, eosD2o, use_IAPWS_st, eosNames, use_D2O_st
!         USE EosD2oSTData, ONLY: stread
         USE Io, ONLY: sTEosFName, sTEosFIn, outUnt !, d2oSTEosFName, d2oSTEosFIn
!
         IMPLICIT NONE
!
!  Purpose: This routine gets the H2O steam tables.  The H2O steam tables could be located in data files:
!           "testProblemName.h2o", "trch2o", or "tpfh2onew".  If none of these files are local, then the ascii
!           default steam tables are loaded.
!           Note this routine must be called after CountFluids has been called.
!
!  Programmed by Jay Spore, ISL, Date (06/05)
!
         INTEGER(sik), INTENT(IN) :: tbl_opt, fid_msg
!
!  Variable Declarations:
!
         INTEGER(sik) :: i, j, npass, nuse, ios
!
         LOGICAL :: water ! If .TRUE., then this model includes light and/or heavy water.
                          ! If this model includes any water, then steam tables must be
                          ! loaded either from the steam table data file or from the
                          ! default steam table data.
         LOGICAL :: found, exists, defaultSTables, d2oST
!
         CHARACTER (LEN = 140) :: message
         CHARACTER (LEN = 80), DIMENSION(2) :: record
!
!        Check to see if there is any water in this model.
!        This logic assumes that CountFluids has been called.
!         water = .FALSE.
         water = .TRUE.
!         d2oST = .FALSE.
!         DO i = 1, nActEos
!!
!!           Check to see if one of the fluids in this model is light or heavy water.
!!           Heavy water (i.e., eosD2o) uses IAPWS fits for thermal conductivity that require some
!!           data from the steam tables.
!!           The TRAC-P fits (i.e., use_IAPWS_st = .FALSE.) uses IAPWS fits for thermal conductivity and
!!           viscosity that require some data the steam tables.
!!           Of course if use_IAPWS_st = .TRUE., then the steam tables must be loaded.
!            SELECT CASE (actFluids(i))
!            CASE (eosH2o, eosD2o)
!               water = .TRUE.
!               EXIT
!            END SELECT
!         END DO
!         IF (use_D2O_st) THEN
!            d2oST = .TRUE.
!!
!!           Check to see if any fluids elements equal d2o.
!            found = .FALSE.
!            DO i = 1, nActEos
!               IF (actFluids(i) == eosD2o) THEN
!                  found = .TRUE.
!                  EXIT
!               END IF
!            END DO
!            IF (.NOT. found) THEN
!               CALL error(2, ' If use_D2O_st is .TRUE., then at least one element in fluids must be d2o.')
!               CALL error(1, ' use_D2O_st is .TRUE., but no element in fluids input equals d2o.')
!            END IF
!         END IF
!!
!!        If no light or heavy water in this model, then there is no need for light water steam tables.
!         IF (.NOT. water) THEN
!            IF (use_IAPWS_st) THEN
!               DO j = 1, SIZE(outUnt)
!                  WRITE (outUnt(j), '(a)') ' *GetSteamTables* no water in this model: &
!                 &USE_IAPWS_ST will be set to false.'
!               END DO
!               use_IAPWS_st = .FALSE.
!            END IF
!         END IF
!!
!!        Check for consistency between use_IAPWS_st and fluids.
!         IF (use_IAPWS_st) THEN
!            found = .FALSE.
!            DO i = 1, nActEos
!               IF (actFluids(i) == eosH2o) THEN
!                  found = .TRUE.
!                  EXIT
!               END IF
!            END DO
!            IF (.NOT. found) THEN
!               CALL error(2, ' If use_IAPWS_st is .TRUE., then at least one element in fluids must be h2o.')
!               CALL error(1, ' use_IAPWS_st is .TRUE., but no element in fluids input equals h2o.')
!            END IF
!         END IF
!!
!!        Check if D2O steam table file is local.
!         IF (d2oST) THEN
!!
!!           Water is .TRUE. and d2o steam tables area also in this input model.
!!           Check to see if d2o steam tables are present.
!            INQUIRE(FILE = TRIM(d2oSTEosFName), EXIST = exists)
!            IF (.NOT. exists) THEN
!               IF (TRIM(d2oSTEosFName) /= 'trcd2o') THEN
!                  INQUIRE(FILE='trcd2o', EXIST = exists)
!                  IF (exists) THEN
!                     d2oSTEosFName = 'trcd2o'
!                     OPEN (UNIT=d2oSTEosFIn, FILE=TRIM(d2oSTEosFName), STATUS='old', FORM='unformatted', IOSTAT=ios, &
!                    & ACTION='read')
!                     IF (ios /= 0) THEN
!                        WRITE (message, '(3a, i0)') ' Failed opening D2O steam tables data file ', TRIM(d2oSTEosFName), &
!                       & ' with IOSTAT = ', ios
!                        CALL error(1, message)
!                     END IF
!                  END IF
!               END IF
!               IF (.NOT. exists) THEN
!!
!!                 File d2oSTEosFName must be present, if d2oST is .TRUE.
!                  CALL error(2, ' Fluids array input implies that D2O steam table is in this input model.')
!                  jflag = 1
!                  WRITE (message, '(3a)') ' D2O steam tables data file ', TRIM(d2oSTEosFName), &
!                 & ' or trcd2o must be present in the local file space.'
!                  CALL error(1, message)
!               END IF
!            ELSE
!!
!!              D2O steam table present, so open and read it.
!               OPEN (UNIT=d2oSTEosFIn, FILE=TRIM(d2oSTEosFName), STATUS='old', FORM='unformatted', IOSTAT=ios, &
!              & ACTION='read')
!               IF (ios /= 0) THEN
!                  WRITE (message, '(3a, i0)') ' Failed opening D2O steam tables data file ', TRIM(d2oSTEosFName), &
!                       & ' with IOSTAT = ', ios
!                  CALL error(1, message)
!               END IF
!            END IF
!!
!!           Read the data file
!            WRITE (message, '(2a)') ' Reading D2O properties from file ', TRIM(d2oSTEosFName)
!            CALL error(2, message)
!            CALL stread(d2oSTEosFIn, nuse, record)
!            IF (nuse < 0) THEN
!               CALL error(2, ' *GetSteamTables* could not read D2O steam tables.')
!               jflag = 1
!            END IF
!            CLOSE(d2oSTEosFIn)
!!
!         END IF
!!
!        Check to see if h2o steam tables are needed.
         IF (water) THEN
!
!           Try and locate the ST EOS data file.  Depending upon how TRACE was invoked, it can either be called
!           trch2o, tpfh2onew or "prefix".h2o.  The prefix.h2o convention is used to allow multiple code runs in
!           the same directory without file locking.  This is necessary for the NRC regression test suite.  To
!           preserve the existing functionality, if "prefix.h2o" does not exist, then search for "trch2o" and
!           then "tpfh2onew".
!
!        Check to see if the default steam table file name is local or not.
!            defaultSTables = .FALSE.
!            INQUIRE(FILE=TRIM(sTEosFName), EXIST=exists)
!            IF (.NOT. exists) THEN
!!
!!           Could not find the default name for the steam tables file.  Default name could be
!!           "trch2o", "tpfh2onew" or "testProblemName.h2o".
!!               DO j = 1, SIZE(outUnt)
!!                  WRITE (outUnt(j), '(a)') ' *GetSteamTables* could not locate IAPWS steam table data file - ' // &
!!                 & TRIM(sTEosFName)
!!               END DO
!               IF (TRIM(sTEosFName) == 'trch2o' .OR. TRIM(sTEosFName) == 'tpfh2onew') THEN
!                  npass = 1_sik
!               ELSE
!                  npass = 2_sik
!               END IF
!!
!!           Search to see if trch2o or tpfh2onew are local.  Note if the first inquire was for "trch2o",
!!           then this do loop will only look for tpfh2onew.
!               found = .FALSE.
!               DO i = 1, npass
!                  IF (TRIM(sTEosFName) == 'trch2o') THEN
!                     sTEosFName = 'tpfh2onew'
!                  ELSE
!                     sTEosFName = 'trch2o'
!                  END IF
!!                  DO j = 1, SIZE(outUnt)
!!                     WRITE (outUnt(j), '(a)') ' *GetSteamTables* Will look for file - ' // TRIM(sTEosFName)
!!                  END DO
!                  INQUIRE(FILE=sTEosFName, EXIST=exists)
!                  IF (exists) THEN
!                     found = .TRUE.
!                     EXIT
!                  ELSE
!!                     DO j = 1, SIZE(outUnt)
!!                        WRITE (outUnt(j), '(a)') ' *GetSteamTables* could not locate IAPWS steam table data file - ' // &
!!                       & TRIM(sTEosFName)
!!                     END DO
!                  END IF
!               END DO
!
               !ADD THIS BACK IN FOR LOADING TABLE FROM TRACE
!           Check to see if a steam tables file was not found. 
               IF (tbl_opt .EQ. 1) THEN
                  sTEosFName = 'tpfh2onew'
                  INQUIRE(FILE=TRIM(sTEosFName), EXIST=exists)
                  IF (exists) THEN
                      defaultSTables = .FALSE.
                  ELSE
                      DO j = 1, SIZE(outUnt)
                          WRITE (outUnt(j), '(a)') ' *GetSteamTables* tpfh2onew file not found'
                      END DO    
                      STOP
                  END IF    
               ELSE IF (tbl_opt .EQ. 0) THEN
                  DO j = 1, SIZE(outUnt)
!                     WRITE (outUnt(j), '(a)') ' *GetSteamTables* Will use default built-in IAPWS steam table data'
!                     WRITE (outUnt(j), '(a)') ' *GetSteamTables* Loading built-in IAPWS steam table data.'
                     WRITE (fid_msg, '(a)') ' *GetSteamTables* Will use default built-in IAPWS steam table data'
                     WRITE (fid_msg, '(a)') ' *GetSteamTables* Loading built-in IAPWS steam table data.'
                  END DO
                  CALL EosLoadIAPWS(fid_msg)
                  defaultSTables = .TRUE.
               ELSE                  
                  DO j = 1, SIZE(outUnt)
                      WRITE (outUnt(j), '(a)') ' *GetSteamTables* option not recognized, check input'
                  END DO                   
                  STOP
               END IF
!            END IF
!
!        If the default ascii steam tables were used, then it is not necessary to open steam table files.
            IF (.NOT. defaultSTables) THEN
!
!           Open the steam table data file.
               DO j = 1, SIZE(outUnt)
                  WRITE (outUnt(j), '(a)') ' *GetSteamTables* Found ' // TRIM(sTEosFName) // &
                    '.  Opening file for read...'
               END DO
               OPEN (UNIT=sTEosFIn, FILE=TRIM(sTEosFName), STATUS='old', FORM='unformatted', IOSTAT=ios, &
              & ACTION='read')
               IF (ios /= 0) THEN
                  DO j = 1, SIZE(outUnt)
                     WRITE (outUnt(j), '(a)') ' *GetSteamTables* - Error while opening IAPWS steam table data file ' &
                    & // TRIM(sTEosFName)
                  END DO
                  STOP
               END IF
!
!           Read the steam table data file.
               CALL newstREAD(sTEosFIn, nuse, fid_msg)
               IF (nuse ==-1) THEN
                  DO j = 1, SIZE(outUnt)
                     WRITE (outUnt(j), '(a)') ' *GetSteamTables* could not read IAPWS steam tables.'
                  END DO
               END IF
               CLOSE(sTEosFIn)
            END IF
         END IF
!
      END SUBROUTINE GetSteamTables
!
END MODULE EosIAPWSRead
