!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_allocM

      USE IntrType, ONLY: sdk, sik
      USE PATHS_varcpld
      USE PATHS_thvarM

      IMPLICIT NONE

CONTAINS

      SUBROUTINE allocpaths
      
            USE PATHS_thvarM

            IMPLICIT NONE
            
            PATHS_nz=nzpl + nchm
!            IF (PATHS_usechimney) PATHS_nz=PATHS_nz+1
            PATHS_nax=PATHS_nz+1
            PATHS_matsize=2*PATHS_nz*PATHS_nchan+1
            PATHS_matnnz=6*PATHS_nz*PATHS_nchan+1
            PATHS_nfm=PATHS_nr
            PATHS_ngm=PATHS_nr+1
            PATHS_ncm=PATHS_nr+3
            PATHS_nasm=nfbxy
! This may be for hex, but new inputs may get around problem. Need to test.            
!            PATHS_nasm=PATHS_nchan  ! Aaron - nfuelfa not defined at this point in the code for hex geom (standalone PATHS)
! nfuel = PATHS_nchan??

! Use in input reader            
            ALLOCATE(PATHS_Prel(PATHS_nz,PATHS_nchan))
            ALLOCATE(PATHS_Preld(PATHS_nz,PATHS_nchan))

            ALLOCATE(PATHS_chantype(PATHS_nchan))

            ALLOCATE(PATHS_hz(PATHS_nz))            
            
            ALLOCATE(PATHS_conf(PATHS_nasm))         
            ALLOCATE(PATHS_ct_map(PATHS_nasm))

            ALLOCATE(PATHS_rgh(PATHS_nchantype))
            ALLOCATE(PATHS_pit(PATHS_nchantype))
            ALLOCATE(PATHS_rrod(PATHS_nchantype))
            ALLOCATE(PATHS_rf(PATHS_nchantype))
            ALLOCATE(PATHS_tc(PATHS_nchantype))
            ALLOCATE(PATHS_kgap(PATHS_nchantype))
            ALLOCATE(PATHS_nwrod(PATHS_nchantype))
            ALLOCATE(PATHS_rwrod(PATHS_nchantype))
            ALLOCATE(PATHS_kloss(PATHS_nz,PATHS_nchantype))            

            ALLOCATE(PATHS_nrod(PATHS_nz,PATHS_nchantype))            
!            ALLOCATE(PATHS_nrod(PATHS_nchantype))
            
! Added for loss and area change            
            ALLOCATE(PATHS_achan(PATHS_nz,PATHS_nchantype))
!            ALLOCATE(PATHS_achan(PATHS_nchantype))
            ALLOCATE(PATHS_dhz(PATHS_nz,PATHS_nchantype))            
!            ALLOCATE(PATHS_dhz(PATHS_nchantype))
            ALLOCATE(PATHS_aloss(PATHS_nz+1,PATHS_nchantype))            

! Used outside of input reader            
            ALLOCATE(PATHS_nasmchan(PATHS_nchan))

! Volume of each node for use in coupling to PARCS
            ALLOCATE(PATHS_hydvol(PATHS_nz,PATHS_nchan))            
            
! Orifice losses            
            ALLOCATE(PATHS_oriftype(PATHS_nchan))
            ALLOCATE(PATHS_orifloss(PATHS_nchan)) ! conservative value for size
            
            ALLOCATE(PATHS_u(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_unew(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_P(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_enth(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_dens(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_densnew(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_qual(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_vf(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_vgj(PATHS_nax,PATHS_nchan))
            ALLOCATE(PATHS_tcool(PATHS_nax,PATHS_nchan))                        
! Added
            ALLOCATE(PATHS_regnm(PATHS_nax,PATHS_nchan))  ! Aaron - added for Liao correlation
!            
            ALLOCATE(PATHS_xcrit(PATHS_nax,PATHS_nchan))  ! Aaron - added for critical quality correlation (CISE)
            ALLOCATE(PATHS_cpr(PATHS_nax,PATHS_nchan))  ! Aaron - added for critical quality correlation (CISE)
            ALLOCATE(PATHS_xeq(PATHS_nax,PATHS_nchan))  ! Aaron - added for critical quality correlation (CISE)
            ALLOCATE(PATHS_boil_len(PATHS_nchan))  ! Aaron - added for critical quality correlation (CISE)

            ALLOCATE(PATHS_tw(PATHS_nz,PATHS_nchan))

!            ALLOCATE(PATHS_tavg(PATHS_ncm,PATHS_nz,PATHS_nchan))
            ALLOCATE(PATHS_tavg(PATHS_ncm+1,PATHS_nz,PATHS_nchan)) ! Store average whole pin
            ALLOCATE(PATHS_tfuel(PATHS_ncm+1,PATHS_nz,PATHS_nchan))
            ALLOCATE(keff(PATHS_ncm))
            ALLOCATE(tavgo(PATHS_ncm))
            ALLOCATE(rfuel(PATHS_ncm))

            ALLOCATE(PATHS_pt(64))
            ALLOCATE(PATHS_iparam(64))
            ALLOCATE(PATHS_matia(PATHS_matsize+1))
            ALLOCATE(PATHS_matja(PATHS_matnnz))
            ALLOCATE(PATHS_mata(PATHS_matnnz))
            ALLOCATE(PATHS_matx(PATHS_matsize))
            ALLOCATE(PATHS_matb(PATHS_matsize))
            
! Added for extraction from PARCS            
            ALLOCATE(relpnew(PATHS_nz,PATHS_nasm))
            
! User specified conductivities
            IF (num_kfuel .GT. 0) THEN
                ALLOCATE(dat_kfuel(num_kfuel))
                dat_kfuel = 0.0
            END IF
            IF (num_kclad .GT. 0) THEN
                ALLOCATE(dat_kclad(num_kclad))
                dat_kclad = 0.0
            END IF
            
            PATHS_Prel = 0.0
            PATHS_Preld = PATHS_Prel
            
            PATHS_chantype = 0
            PATHS_achan = 0.0

      END SUBROUTINE allocpaths

END MODULE PATHS_allocM
