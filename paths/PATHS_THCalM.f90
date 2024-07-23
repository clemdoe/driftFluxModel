!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_THCal

      USE IntrType, ONLY: sdk, sik
      USE PATHS_util, ONLY: mesg, term_execution
      USE PATHS_thvarM
      USE PATHS_blas1
      USE PATHS_blas2
      USE PATHS_bicgstab
      USE PATHS_linsolve
      USE PATHS_varcpld, ONLY: plevnew, relpnew, get_blnk_unit

!      USE GeomM
!      USE GeomhfcM, ONLY: isymang
!      USE PowerM
!      USE ThgeomM, ONLY: nchan, nzth
!      USE DEP_IO_var, ONLY: ncolumn
!      USE CntlM, ONLY: popt       
      
      IMPLICIT NONE

      ! contains the standalone TH calculation routines

CONTAINS

      SUBROUTINE PATHS_iniTH

            USE PATHS_WaterProp, ONLY: rhof, rhog, hf, t_to_enth
            USE PATHS_thmodels
            USE PATHS_ftcalc,  ONLY:init_tf    
            USE PATHS_thvarM, ONLY: stable
            USE EosIAPWSRead, ONLY: GetSteamTables
            USE EosInit            
            
            IMPLICIT NONE

            LOGICAL :: in_use
            
            INTEGER(sik)::i, j, k
            REAL(sdk)::Pnode, rhou, relpsum, klosssum
            ! PATHS version 2.0 revision
!            CHARACTER(len=40) :: infile, outfile, parcsfile, version
!            CHARACTER(len=1) :: BLANK
            CHARACTER(len=100) :: CARDN,filen
            CHARACTER(len=2) :: chan_id
            REAL(sdk) :: multi 
            INTEGER(sik):: lfas, lfae, kt, kk, lfa
!            grav=9.80665           !m/s**2
            grav=9.81
            
            IF (fid_pth .EQ. 0) THEN
                filen=TRIM(paths_id)//'.paths_pth'
                CALL get_blnk_unit(fid_pth)
                OPEN(fid_pth,file=filen, status='unknown')                
            ELSE    
                INQUIRE(fid_pth,OPENED=in_use)
                IF (.NOT. in_use) THEN
                    filen=TRIM(paths_id)//'.paths_pth'
                    CALL get_blnk_unit(fid_pth)
                    OPEN(fid_pth,file=filen, status='unknown')
                END IF
            END IF    
    
            IF (fid_xth .EQ. 0) THEN
                filen=TRIM(paths_id)//'.paths_xth'
                CALL get_blnk_unit(fid_xth)
                OPEN(fid_xth,file=filen, status='unknown')
            ELSE    
                INQUIRE(fid_xth,OPENED=in_use)
                IF (.NOT. in_use) THEN
                    filen=TRIM(paths_id)//'.paths_xth'
                    CALL get_blnk_unit(fid_xth)
                    OPEN(fid_xth,file=filen, status='unknown')
                END IF
            END IF            
            
            PATHS_mhtc=1
            PATHS_mfs=1
            PATHS_mdb=2

!            PATHS_hz(1:PATHS_nz)=hz(1:PATHS_nz)/100.0
!            PATHS_hz(1:PATHS_nz)=PATHS_hz(1:PATHS_nz)/100.0

! Assign chimney height to hz array
            IF (PATHS_usechimney) PATHS_hz(PATHS_nz) = PATHS_hzchimney/100.0 ! Aaron - top node is chimney

            DO i=1,PATHS_nchan
! Ensure area data was given for all channels used                
               k=PATHS_chantype(i) 
               IF (PATHS_achan(1,k) .LE. 0.0) THEN
                   WRITE(chan_id,'(I2)') k
                   mesg = 'Error: Channel type '//chan_id//' was used, but area data missing'
                   CALL term_execution(mesg)
               END IF    
! zero out regnm array                           
               DO j=1,PATHS_nax
                  PATHS_regnm(j,i)=0.0
               ENDDO
            ENDDO
            
! correct symmetry for partial radial model            
            IF(coor .EQ. 3) THEN
               PATHS_symmmult = isymang/360.0
            ELSEIF(coor .EQ. 1) THEN
               PATHS_symmmult = 1.0/isymmetry
            ENDIF
            
            
! Read in steam tables
            IF(stable) THEN
               CALL GetSteamTables(stm_opt, fid_out)
               CALL SetEos
               CALL MinMaxPT
            ENDIF            
            
            !Initialization of variables and initial guesses
            DO i=1,PATHS_nchan
               PATHS_nasmchan(i)=COUNT(PATHS_conf.EQ.i)
            ENDDO

            !++++++++++++++++++++++++++++++++++++ New matrix A assignment +++++++++++++++++++++++++++
            !Mass Conservation 2 entries per line in the second nz row of each channal matrix
            DO i=1,PATHS_nchan
               DO j=1,PATHS_nz
                  PATHS_matia(PATHS_nz*(2*i-1)+j)=(6*PATHS_nz-1)*i-2*PATHS_nz+2*j-1
                  PATHS_matja((6*PATHS_nz-1)*i-2*PATHS_nz+2*j-1)=PATHS_nz*(2*i-1)+j-1
                  PATHS_matja((6*PATHS_nz-1)*i-2*PATHS_nz+2*j)=PATHS_nz*(2*i-1)+j                       
               ENDDO
            ENDDO

            !Momentum Conservation 3 entries in the first line of each channal matrix
            !Momentum Conservation 4 entries per line in the 2 to nz row of each channal matrix
            DO i=1,PATHS_nchan
               PATHS_matia(2*PATHS_nz*(i-1)+1)=(6*PATHS_nz-1)*(i-1)+1
               PATHS_matja((6*PATHS_nz-1)*(i-1)+1)=2*PATHS_nz*(i-1)+1
               PATHS_matja((6*PATHS_nz-1)*(i-1)+2)=2*PATHS_nz*(i-1)+2*PATHS_nz-1
               PATHS_matja((6*PATHS_nz-1)*(i-1)+3)=2*PATHS_nz*(i-1)+2*PATHS_nz
               DO j=2,PATHS_nz-1
                  PATHS_matia(2*PATHS_nz*(i-1)+j)=(6*PATHS_nz-1)*(i-1)+4*(j-1)
                  PATHS_matja((6*PATHS_nz-1)*(i-1)+4*j-4)=2*PATHS_nz*(i-1)+j-1
                  PATHS_matja((6*PATHS_nz-1)*(i-1)+4*j-3)=2*PATHS_nz*(i-1)+j
                  PATHS_matja((6*PATHS_nz-1)*(i-1)+4*j-2)=2*PATHS_nz*(i-1)+2*PATHS_nz-j 
                  PATHS_matja((6*PATHS_nz-1)*(i-1)+4*j-1)=2*PATHS_nz*(i-1)+2*PATHS_nz-j+1
               ENDDO
               PATHS_matia(PATHS_nz*(2*i-1))=(6*PATHS_nz-1)*(i-1)+4*(PATHS_nz-1)
               PATHS_matja((6*PATHS_nz-1)*(i-1)+4*PATHS_nz-4)=2*PATHS_nz*(i-1)+PATHS_nz-1
               PATHS_matja((6*PATHS_nz-1)*(i-1)+4*PATHS_nz-3)=2*PATHS_nz*(i-1)+PATHS_nz
               PATHS_matja((6*PATHS_nz-1)*(i-1)+4*PATHS_nz-2)=2*PATHS_nz*(i-1)+PATHS_nz+1
               PATHS_matja((6*PATHS_nz-1)*(i-1)+4*PATHS_nz-1)=2*PATHS_nz*PATHS_nchan+1
            ENDDO
            PATHS_matia(PATHS_matsize)=(6*PATHS_nz-1)*PATHS_nchan+1
            PATHS_matia(PATHS_matsize+1)=PATHS_matnnz+1

            !nz entries in the last row 
            DO i=1,PATHS_nchan
               PATHS_matja((6*PATHS_nz-1)*PATHS_nchan+i)=2*PATHS_nz*(i-1)+PATHS_nz
            ENDDO

            !the last entry in the last row
            PATHS_matja(PATHS_matnnz)=PATHS_matsize
            !++++++++++++++++++++++++++++++++++++ New matrix A assignment +++++++++++++++++++++++++++ 


            ! Initialize solution vector to zero
            DO i=1,PATHS_matsize
               PATHS_matb(i)=0.0
               PATHS_matx(i)=0.0
            ENDDO
            DO i=1,PATHS_matnnz
               PATHS_mata(i)=0.0
            ENDDO            

            ! Initialize parameters and solve energy equation
            DO i=1,PATHS_nchan
               DO j=1,PATHS_nax
                  PATHS_P(j,i)=PATHS_Pout
                  PATHS_enth(j,i)=PATHS_hin
                  PATHS_dens(j,i)=rhof(PATHS_P(j,i), j, i, .TRUE.)
                  IF(j<PATHS_nax) THEN
                    PATHS_u(j,i)=PATHS_mdotcore*PATHS_symmmult/PATHS_nasm/(PATHS_dens(j,i)*PATHS_achan(j,PATHS_chantype(i)))
                  ELSE
                    PATHS_u(j,i)=PATHS_mdotcore*PATHS_symmmult/PATHS_nasm/(PATHS_dens(j,i)*PATHS_achan(j-1,PATHS_chantype(i)))
                  ENDIF
                  PATHS_unew(j,i)=PATHS_u(j,i)
                  PATHS_qual(j,i)=0.0
               ENDDO
            ENDDO
            
            IF(.NOT.freezedc) THEN
               CALL init_tf
            ENDIF
            
            !Debugging
            IF(PATHS_debug .GT. 0) THEN
                filen=TRIM(paths_id)//'.paths_itrerr'
                CALL get_blnk_unit(fid_tmp1)
                OPEN(unit=fid_tmp1,file=filen,status='unknown')
                WRITE(fid_tmp1,*) '     # inners   final inner error     current outer error'
            ENDIF

            !Verbosity
            IF(PATHS_verbos .GT. 0) THEN
                !filen=TRIM(paths_id)//'.chk_relp'
                !CALL get_blnk_unit(fid_tmp2)
                !OPEN(unit=fid_tmp2,file=filen,status='unknown')
                filen=TRIM(paths_id)//'.paths_tfuel'
                CALL get_blnk_unit(fid_tmp3)
                OPEN(unit=fid_tmp3,file=filen,status='unknown')
                !filen=TRIM(paths_id)//'.var_parm'
                !CALL get_blnk_unit(fid_tmp4)
                !OPEN(unit=fid_tmp4,file=filen,status='unknown')
                filen=TRIM(paths_id)//'.paths_relp'
                CALL get_blnk_unit(fid_tmp5)
                OPEN(unit=fid_tmp5,file=filen,status='unknown')
                filen=TRIM(paths_id)//'.paths_flow'
                CALL get_blnk_unit(fid_tmp6)
                OPEN(unit=fid_tmp6,file=filen,status='unknown')
            ENDIF

            !CPR
            IF(PATHS_cpropt .GT. 0) THEN
                filen=TRIM(paths_id)//'.paths_qual'
                CALL get_blnk_unit(fid_tmp7)
                OPEN(unit=fid_tmp7,file=filen,status='unknown')
                filen=TRIM(paths_id)//'.paths_cpr'
                CALL get_blnk_unit(fid_tmp8)
                OPEN(unit=fid_tmp8,file=filen,status='unknown')
            ENDIF
            
      END SUBROUTINE PATHS_iniTH

      SUBROUTINE PATHS_UpdatePower

            USE PATHS_WaterProp, ONLY: rhof, rhog, hf
            USE PATHS_thmodels
            USE PATHS_util, ONLY: nfields            

            IMPLICIT NONE

            INTEGER(sik), PARAMETER      ::   ncolumn=10
            INTEGER(sik)::i, j, k
            REAL(sdk)::Pnode, rhou, relpsum, klosssum
!            CHARACTER(len=40) :: infile, outfile, parcsfile, version
            CHARACTER(len=100) :: CARDN, form
            REAL(sdk) :: multi ,maxcPr,Pr,del,Prel_sum
            INTEGER(sik):: kt, kk, ncol, io_flg
            LOGICAL :: io_open

            INTEGER(sik) :: fa,cnt
            INTEGER(sik) :: tmpread(10)
            REAL(sdk) :: tmpdata(10)

            ! Store previous
            PATHS_Preld = PATHS_Prel                            
            
            ! Grab power shape if avaiable from user inputs for first iteration
            IF (first_solve) THEN
                ! Input a dep file
                IF(pwr_inp .EQ. 2) THEN
                    INQUIRE(unit = fid_rlp, OPENED=io_open, IOSTAT=io_flg)
                    IF (io_open) CLOSE(fid_rlp)
                    CALL get_blnk_unit(fid_rlp)
                    OPEN(fid_rlp,file=PATHS_depfname,form='FORMATTED')                
                    DO 
                        READ(fid_rlp,'(1x,a8)',END=100)CARDN
                        IF(CARDN.EQ.'RPF 3D M') THEN
                           BACKSPACE(fid_rlp)
                           multi = 1.0_sdk
                           READ(fid_rlp,'(12x,E7.1)') multi
                           IF( multi.EQ.0.0d0) multi=1.0d0
                           cnt = 0
                           relpnew = 0.0
                           srch: DO WHILE(.TRUE.)
                               READ(fid_rlp,*)
                               READ(fid_rlp,'(a1024)') oneline
                               ncol=nfields(oneline)-2
                               DO kk=nzpl,1,-1 ! Need to be specified for reflector planes
                                   READ(fid_rlp,*)kt,tmpdata(1:ncol)
                                   IF (kk .EQ. nzpl .AND. kt.NE.kk) THEN
                                       mesg = 'Error reading power from depletion file: wrong number of axial points.'
                                       CALL term_execution(mesg)
                                       ! Aaron - figure out proper way to give error
                                   ENDIF
                                   IF (kk .EQ. nzpl-nzur) THEN
                                        tmpread = 0
                                        DO fa=1,10
                                            IF (tmpdata(fa) .GT. 0) THEN
                                                cnt = cnt + 1
                                                tmpread(fa) = cnt
                                            END IF
                                        END DO
                                   END IF
                                   DO fa=1,ncol
                                        IF (tmpread(fa) .GT. 0) THEN
                                            relpnew(kk, tmpread(fa)) = tmpdata(fa)
                                        END IF
                                   END DO
                               ENDDO
                               IF (cnt .EQ. PATHS_nchan)THEN
                                   EXIT srch
                               ENDIF
                           END DO srch
                        
! Check between     relpnew and relp                     
                          IF(multi.NE.1.0d0)relpnew=relpnew*multi
                       ELSEIF(CARDN=='END STEP') THEN
                          EXIT                        
                       ENDIF
                    ENDDO
                    relpsum=0
                    DO i=1,PATHS_nasm
                       DO j=1,nzpl
                          IF(relpnew(j,i).LT.0.0)relpnew(j,i)=0.0
                          relpsum=relpsum+relpnew(j,i)*PATHS_hz(j)
                       ENDDO
                    ENDDO
                    DO i=1,PATHS_nchan
                       DO j=1,PATHS_nz
                          Pr=0
                          Pr=Pr+relpnew(j,i)
                          Pr=Pr/relpsum/REAL(PATHS_nasmchan(i))*REAL(PATHS_nasm)
                          PATHS_Prel(j,i)=pr                         
                       ENDDO
                    ENDDO                   
                ELSE IF (pwr_inp .EQ. 1) THEN
                    ! Taken directly from input
                    Prel_sum=0.0_sdk
                    DO i=1,PATHS_nchan
                       DO j=1,PATHS_nz
                          Prel_sum = Prel_sum+PATHS_Prel(j,i)*PATHS_hz(j)
                       ENDDO
                    ENDDO
                    PATHS_Prel = PATHS_Prel/Prel_sum*PATHS_nchan
                ELSE
                    relpnew = 0.0 
                    relpsum = 0.0
                    DO i=1,PATHS_nasm
                        DO j=nzlr + 1,nzpl - nzur
                            relpnew(j,i)=1.0
                            relpsum=relpsum+relpnew(j,i)*PATHS_hz(j)
                        ENDDO
                    ENDDO
                    DO i=1,PATHS_nchan
                        DO j=1,PATHS_nz
                            Pr=0
                            Pr=Pr+relpnew(j,i)
                            Pr=Pr/relpsum/REAL(PATHS_nasmchan(i))*REAL(PATHS_nasm)
                            PATHS_Prel(j,i)=pr                         
                        ENDDO
                    ENDDO                    
                ENDIF
            ELSE
               relpsum=0
               DO i=1,PATHS_nasm
                  DO j=1,nzpl
                     IF(relpnew(j,i).LT.0.0) THEN
                        relpnew(j,i)=0.0
                     ENDIF
                     relpsum=relpsum+relpnew(j,i)*PATHS_hz(j)
                  ENDDO
               ENDDO
               maxcPr=0
               DO i=1,PATHS_nchan
                  DO j=1,PATHS_nz
                     Pr=0
                     Pr=Pr+relpnew(j,i)
                     Pr=Pr/relpsum/REAL(PATHS_nasmchan(i))*REAL(PATHS_nasm)
                     del=ABS(PATHS_Prel(j,i)-Pr)
                     IF(maxcPr<del)maxcPr=del
                     IF (PATHS_Preld(j,i) .GT. 0.0) THEN
                         PATHS_Prel(j,i)=pow_rlx * pr + (1.0 - pow_rlx) * PATHS_Preld(j,i)
                     ELSE
                         PATHS_Prel(j,i)=pr                         
                     END IF
                  ENDDO
               ENDDO
            ENDIF

            IF (PATHS_usechimney) PATHS_Prel(PATHS_nz,i) = 0.0_sdk                              
            
            ! Initialize solution vector to ze;ro
            DO i=1,PATHS_matsize
               PATHS_matb(i)=0.0
               PATHS_matx(i)=0.0
            ENDDO
            PATHS_matb(PATHS_matsize)=PATHS_mdotcore*PATHS_symmmult

            RETURN
100         WRITE(fid_out,*)'PATHS error: not enough points in restart file!'
            
      END SUBROUTINE PATHS_UpdatePower

      SUBROUTINE PATHS_ssthiter

            USE PATHS_WaterProp, ONLY: rhof, rhog, hf, t_to_enth
            USE PATHS_thmodels
            IMPLICIT NONE

            INTEGER(sik)::i, j, k, itrmx, nout
            REAL(sdk)::Pnode, efffrict, error, epscrt, relpsum, del, maxcV, maxcD, maxcP, maxcE,temp,temp_nonlin
            REAL(sdk)::reg1_hz,reg2_hz
            INTEGER(sik) :: reg1, reg2, j2 
!            ! Finds edges where area changes (will add special dp term at these edges)
!            k=0
!            DO i=1,SIZE(PATHS_chantype)
!               DO j=2,PATHS_nz
!                  IF(PATHS_achan(j-1,i).NE.PATHS_achan(j,i)) THEN
!                     k=k+1
!                     PATHS_achg(k,i)=j
!                  ENDIF
!               ENDDO
!               PATHS_numachg(i)=k
!            ENDDO

            !++++++++++++++++++++++++++++++++++++ New matrix A assignment +++++++++++++++++++++++++++++++++++++

            ! Aaron 10/12 - added variable flow area - abrupt area changes only, located at cell edges
            !         Abrupt area change creates a discontinuity in velocity and pressure at the cell edge (differing on the
            !         'plus' and 'minus' side of each edge.  The following convention is used here:
            !         Areas are input and defined on the 'plus' side of each edge and apply to entire node beyond that edge.
            !         All velocities solved in the matrix and outputted are also defined on the 'plus' side of each edge;
            !         when integrating over a cell you get velocities on the 'minus' side too, so an area ratio is used internally
            !         to convert these in terms of the velocities of the 'plus' side.
            !         Note that the last edge (PATHS_nz+1) uses the convention that PATHS_u(PATHS_nz+1)+ = PATHS_u(PATHS_nz+1)-

            
            DO i=1,PATHS_nchan
               ! Core Continuity
               PATHS_mata((6*PATHS_nz-1)*PATHS_nchan+i)=PATHS_nasmchan(i)*PATHS_dens(1,i)*PATHS_achan(1,PATHS_chantype(i))
               DO j=1,PATHS_nz
                  ! Continuity
                  PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz+2*j-1)=-PATHS_dens(j,i)*PATHS_achan(j,PATHS_chantype(i))
                  IF(j.NE.PATHS_nz) THEN
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz+2*j)=PATHS_dens(j+1,i)*PATHS_achan(j+1,PATHS_chantype(i))
                  ELSE
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz+2*j)=PATHS_dens(j+1,i)*PATHS_achan(j,PATHS_chantype(i))
                  ENDIF
                  ! Momentum (Convection)
                  efffrict=0.25*(frict(j,i,1)*tpfmult(j,i,1)/PATHS_dhz(j,PATHS_chantype(i))                          &
                       +PATHS_kloss(j,PATHS_chantype(i))/PATHS_hz(j))                                              &
                       *PATHS_hz(j)*PATHS_achan(j,PATHS_chantype(i))
                  IF(j.EQ.1) THEN
                     IF(PATHS_orifopt.EQ.1) THEN
                        efffrict=efffrict+0.25*PATHS_orifloss(PATHS_oriftype(i))*PATHS_achan(j,PATHS_chantype(i))  
                        ! Add optional orifice loss factor to j=1 node
                     ELSEIF(PATHS_orifopt.EQ.2) THEN
                        
                     ENDIF
                  ENDIF                  
                  IF (j.NE.1)THEN
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+3)=-PATHS_dens(j,i)*ABS(PATHS_u(j,i))          &
                          *(PATHS_achan(j,PATHS_chantype(i))-efffrict)
                  ELSE 
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+2)=-PATHS_dens(j,i)*ABS(PATHS_u(j,i))          &
                          *(PATHS_achan(j,PATHS_chantype(i))-efffrict)
                  ENDIF
                  efffrict=0.25*(frict(j+1,i,-1)*tpfmult(j+1,i,-1)/PATHS_dhz(j,PATHS_chantype(i))                     &
                       +PATHS_kloss(j,PATHS_chantype(i))/PATHS_hz(j))                                            &
                       *PATHS_hz(j)*PATHS_achan(j,PATHS_chantype(i))
                  IF(j.EQ.1) THEN
                     IF(PATHS_orifopt.EQ.1) THEN
                        efffrict=efffrict+0.25*PATHS_orifloss(PATHS_oriftype(i))*PATHS_achan(j,PATHS_chantype(i))  
                        ! Add optional orifice loss factor to j=1 node
                     ELSEIF(PATHS_orifopt.EQ.2) THEN
                        
                     ENDIF
                  ENDIF                  
                  IF (j.NE.1)THEN
                     IF (j.NE.PATHS_nz) THEN
                        PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+4)=PATHS_dens(j+1,i)*ABS(PATHS_u(j+1,i))      &
                             *(PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i)))**2         &
                             *(PATHS_achan(j,PATHS_chantype(i))+efffrict)
                     ELSE
                        PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+4)=PATHS_dens(j+1,i)*ABS(PATHS_u(j+1,i))      &
                             *(PATHS_achan(j,PATHS_chantype(i))+efffrict)
                     ENDIF
                  ELSE
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+3)=PATHS_dens(j+1,i)*ABS(PATHS_u(j+1,i))      &
                           *(PATHS_achan(j,PATHS_chantype(i))+efffrict)
                  ENDIF
                  ! Momentum (Pressure)
                  IF (j.NE.1) THEN
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+2)=-PATHS_achan(j,PATHS_chantype(i))
                     IF (j.NE.PATHS_nz) THEN
                        PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+1)=PATHS_achan(j,PATHS_chantype(i))
                     ENDIF
                  ELSE
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+4)=-PATHS_achan(j,PATHS_chantype(i))
                     PATHS_mata((6*PATHS_nz-1)*i-2*PATHS_nz-4*j+1)=PATHS_achan(j,PATHS_chantype(i))
                  ENDIF
                  ! Momentum Source Term
                  IF (j.NE.PATHS_nz) THEN
                     IF (PATHS_aopt.EQ.0) THEN
                        PATHS_matb(PATHS_nz*(2*i-1)-(j-1))=-0.5*(PATHS_dens(j,i)+PATHS_dens(j+1,i))               &
                              *grav*PATHS_achan(j,PATHS_chantype(i))*PATHS_hz(j)                                   &
                             +PATHS_achan(j,PATHS_chantype(i))*                                                    &
                               PATHS_dens(j+1,PATHS_chantype(i))*PATHS_u(j+1,PATHS_chantype(i))**2/2               &
                               *((PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i)))**2-1)       &
                             +PATHS_achan(j,PATHS_chantype(i))*                                                   &
                              PATHS_dens(j+1,PATHS_chantype(i))*PATHS_u(j+1,PATHS_chantype(i))**2/2               &
                              *(-PATHS_aloss(j+1,PATHS_chantype(i)))
                     ELSEIF (PATHS_aopt.EQ.1) THEN
                        IF (PATHS_achan(j+1,PATHS_chantype(i)) >= PATHS_achan(j,PATHS_chantype(i))) THEN   ! Flow expansion
                           PATHS_matb(PATHS_nz*(2*i-1)-(j-1))=-0.5*(PATHS_dens(j,i)+PATHS_dens(j+1,i))               &
                                 *grav*PATHS_achan(j,PATHS_chantype(i))*PATHS_hz(j)                                   &
                                +PATHS_achan(j,PATHS_chantype(i))                                                    &
                                 *PATHS_dens(j+1,PATHS_chantype(i))*PATHS_u(j+1,PATHS_chantype(i))**2               &
                                 *(PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i))-1)         &
                                +PATHS_achan(j,PATHS_chantype(i))                                                &
                                 *PATHS_dens(j+1,PATHS_chantype(i))*PATHS_u(j+1,PATHS_chantype(i))**2/2               &
                                 *(-PATHS_aloss(j+1,PATHS_chantype(i)))
                        ELSE  ! Flow contraction
                           PATHS_matb(PATHS_nz*(2*i-1)-(j-1))=-0.5*(PATHS_dens(j,i)+PATHS_dens(j+1,i))               &
                                 *grav*PATHS_achan(j,PATHS_chantype(i))*PATHS_hz(j)                                   &
                                +PATHS_achan(j,PATHS_chantype(i))                                                    &
                                 *PATHS_dens(j+1,PATHS_chantype(i))*PATHS_u(j+1,PATHS_chantype(i))**2/2               &
                                 *(-1.5+0.7*PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i))         &
                                 +0.8*(PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i)))**2)     &
                                +PATHS_achan(j,PATHS_chantype(i))                                                &
                                 *PATHS_dens(j+1,PATHS_chantype(i))*PATHS_u(j+1,PATHS_chantype(i))**2/2               &
                                 *(-PATHS_aloss(j+1,PATHS_chantype(i)))
                        ENDIF
                     ENDIF
                  ELSE
                     PATHS_matb(PATHS_nz*(2*i-1)-(j-1))=-0.5*(PATHS_dens(j,i)+PATHS_dens(j+1,i))               &
                          *grav*PATHS_achan(j,PATHS_chantype(i))*PATHS_hz(j)                                   
                  ENDIF
               ENDDO
               PATHS_matb(PATHS_nz*(2*i-1)-(PATHS_nz-1))=PATHS_matb(PATHS_nz*(2*i-1)-(PATHS_nz-1))          &
                    -PATHS_Pout*PATHS_achan(PATHS_nz,PATHS_chantype(i))
            ENDDO
            !the last entry in the last row - 0
            PATHS_mata(PATHS_matnnz)=0.0
            !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            PATHS_err = thnonlintol()
            
            !++++++++++++++++++++++++++++++++++++ Call bicgstab solver ++++++++++++++++++++++++++++++++++++++++
            !epscrt=1.0e-6 
            ! Linsen: loosen the convergence criterion
            !epscrt=1.0e-3
            !Linsen: PATHS_epsin & PATHS_ninmax will be used
            !itrmx=100000
            
            SELECT CASE(thsolver)
               CASE(1)
                  CALL solve_bicg(PATHS_epsin,PATHS_ninmax,PATHS_matsize,PATHS_mata,PATHS_matia,PATHS_matja     &
                       ,PATHS_matb,PATHS_matx,error,nout)
               CASE(2)
                  CALL solve_gs(PATHS_epsin,PATHS_ninmax,PATHS_matsize,PATHS_mata,PATHS_matia,PATHS_matja     &
                       ,PATHS_matb,PATHS_matx,error,nout)
               END SELECT
               
            IF(PATHS_debug .GT. 0) THEN   
               WRITE(fid_tmp1,*) nout, error, PATHS_err
               IF (error .GT. PATHS_epsin) THEN
                  WRITE(fid_tmp1,*) 'Matrix did not converge'
               ENDIF
            ENDIF

            ! Update Velocity and Pressures
            maxcP=ABS(PATHS_P(1,1)-PATHS_matx(PATHS_matsize))
            DO i=1,PATHS_nchan
               DO j=1,PATHS_nax
                  PATHS_unew(j,i)=PATHS_matx(2*PATHS_nz*(i-1)+PATHS_nz+j-1)
                  IF ((j .NE. 1) .AND. (j .NE. PATHS_nax)) THEN
                     temp=PATHS_matx(2*PATHS_nz*(i-1)+PATHS_nz-j+1)
                     del=ABS(PATHS_P(j,i)-temp)
                     IF(maxcP<del)maxcP=del
                     PATHS_P(j,i)=temp
                  ENDIF
               ENDDO
               PATHS_P(1,i)=PATHS_matx(PATHS_matsize)
               PATHS_P(PATHS_nax,i)=PATHS_Pout
            ENDDO
            
            ! Energy Solve
            DO i=1,PATHS_nchan
               j=1
               PATHS_enth(j,i)=PATHS_hin
               PATHS_qual(j,i)=quality(j,i)

               CALL voidfract(j,i)
               PATHS_densnew(j,i)=PATHS_vf(j,i)*rhog(PATHS_P(j,i))+(1.-PATHS_vf(j,i))*rhof(PATHS_P(j,i), j, i, .TRUE.)

               DO j=2,PATHS_nax
                  IF (j<PATHS_nax) THEN
                     Pnode=PATHS_Prel(j-1,i)*PATHS_hz(j-1)*plevnew*PATHS_RTP*PATHS_symmmult   &
                          *(1.0-PATHS_fracdwr-PATHS_fracdvb)/REAL(PATHS_nasm)+(PATHS_unew(j-1,i)   &
                          +PATHS_unew(j,i)*PATHS_achan(j,PATHS_chantype(i))/PATHS_achan(j-1,PATHS_chantype(i)))  &
                          /2000.0*(PATHS_P(j,i)-PATHS_P(j-1,i)-PATHS_dens(j,i)*PATHS_unew(j,i)**2/2        &
                          *(1-(PATHS_achan(j,PATHS_chantype(i))/PATHS_achan(j-1,PATHS_chantype(i)))**2))    &
                          *PATHS_achan(j-1,PATHS_chantype(i))
                     PATHS_enth(j,i)=(PATHS_dens(j-1,i)*PATHS_unew(j-1,i)*PATHS_enth(j-1,i)         &
                          *PATHS_achan(j-1,PATHS_chantype(i))+Pnode)/(PATHS_dens(j,i)*PATHS_unew(j,i)      &
                          *PATHS_achan(j,PATHS_chantype(i)))
                  ELSE
                     Pnode=PATHS_Prel(j-1,i)*PATHS_hz(j-1)*plevnew*PATHS_RTP*PATHS_symmmult   &
                          *(1.0-PATHS_fracdwr-PATHS_fracdvb)/REAL(PATHS_nasm)+(PATHS_unew(j-1,i)   &
                          +PATHS_unew(j,i))  &
                          /2000.0*(PATHS_P(j,i)-PATHS_P(j-1,i)-PATHS_dens(j,i)*PATHS_unew(j,i)**2/2)        &
                          *PATHS_achan(j-1,PATHS_chantype(i))
                     PATHS_enth(j,i)=(PATHS_dens(j-1,i)*PATHS_unew(j-1,i)*PATHS_enth(j-1,i)         &
                          *PATHS_achan(j-1,PATHS_chantype(i))+Pnode)/(PATHS_dens(j,i)*PATHS_unew(j,i)      &
                          *PATHS_achan(j-1,PATHS_chantype(i)))
                  ENDIF
                  PATHS_qual(j,i)=quality(j,i)
                  CALL voidfract(j,i)
                  PATHS_densnew(j,i)=PATHS_vf(j,i)*rhog(PATHS_P(j,i))+(1.-PATHS_vf(j,i))*rhof(PATHS_P(j,i), j, i, .TRUE.)
               ENDDO
            ENDDO
            
            ! Mod for Liao correlation (RBWR) - to smooth the void profile
            IF(PATHS_mvd.EQ.7) THEN
               DO i=1,PATHS_nchan
                  j=0
                  reg1=0
                  reg2=0
                  IF(COUNT(PATHS_regnm(:,i).EQ.(2.5)) > 0) THEN
                     DO WHILE (reg1.EQ.0)
                        j=j+1
                        IF (PATHS_regnm(j,i).EQ.(2.5) .AND. j>1) reg1 = j-1
                     ENDDO
                     DO WHILE (reg2.EQ.0)
                        j=j+1
                        IF (j > PATHS_nax) THEN
                           reg2=PATHS_nax
                           EXIT
                        ENDIF
                        IF (PATHS_regnm(j,i).NE.(2.5)) reg2 = j
                     ENDDO
                     
                     DO j=reg1+1,reg2-1
                        reg1_hz=0
                        reg2_hz=0
                        DO j2=reg1,j-1
                           reg1_hz = reg1_hz + PATHS_hz(j2)
                        ENDDO
                        DO j2=j,reg2-1
                           reg2_hz = reg2_hz + PATHS_hz(j2)
                        ENDDO
                        PATHS_vf(j,i) = (reg1_hz*PATHS_vf(reg2,i)+reg2_hz*PATHS_vf(reg1,i))/(reg1_hz+reg2_hz)
                     ENDDO
                  ENDIF
               ENDDO
            ENDIF
            maxcV=0
            maxcD=0
            DO i=1,PATHS_nchan
               DO j=1,PATHS_nax
                  del=PATHS_unew(j,i)-PATHS_u(j,i)
                  PATHS_u(j,i)=PATHS_u(j,i)+PATHS_thur*del
                  del=ABS(del)
                  IF(maxcV<del)maxcV=del
                  del=PATHS_densnew(j,i)-PATHS_dens(j,i)
                  PATHS_dens(j,i)=PATHS_dens(j,i)+PATHS_thur*del
                  del=ABS(del)
                  IF(maxcD<del)maxcD=del
               ENDDO
            ENDDO
            !IF(PATHS_debug)THEN
                !WRITE(fid_out,'(A,1x, 1pE9.3,1x, A,1x,1pE9.3,1x, A,1x,1pE9.3) ') 'max changes: P ', maxcP,'V ', maxcV,'D', maxcV
            !ENDIF
      END SUBROUTINE PATHS_ssthiter
     
      FUNCTION thnonlintol()

            USE PATHS_thmodels

            IMPLICIT NONE

            INTEGER(sik) :: i,j            
            REAL(sdk) :: err, thnonlintol, efffrictin, efffrictout

            !Aaron 5/13 - generalized the calculation (so you don't have to change it each time you change the solver)
            ! *But you have to call the thnonlintol function just after mata is initialized each iteration, and before
            !  PATHS solves the linear system, so that the velocities and pressures are all the new-time values
            thnonlintol=0;
            DO i=1,PATHS_matsize
                err=PATHS_matb(i)
                DO j=PATHS_matia(i),PATHS_matia(i+1)-1
                    err=err-PATHS_mata(j)*PATHS_matx(PATHS_matja(j))
                ENDDO
                thnonlintol=thnonlintol+err**2
            ENDDO
            thnonlintol=SQRT(thnonlintol)/PATHS_matsize

      END FUNCTION thnonlintol

      FUNCTION thlintol()

            USE PATHS_thmodels

            IMPLICIT NONE

            INTEGER(sik) :: i,j            
            REAL(sdk) :: err, thlintol

            thlintol=0
            DO i=1,PATHS_nchan
               DO j=1,PATHS_nz
                  err=PATHS_unew(j,i)-PATHS_u(j,i)
                  thlintol=thlintol+err**2
               ENDDO
            ENDDO

            thlintol=SQRT(thlintol)

      END FUNCTION thlintol

END MODULE PATHS_THCal
