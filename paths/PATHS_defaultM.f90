!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
SUBROUTINE PATHS_default
      ! --------------------------------------------------------------------------- !
      !           default parameters for PATHS
      ! --------------------------------------------------------------------------- !
      !
      ! Main part of subroutine --------------------------------------------------- !

      USE IntrType, ONLY: sdk, sik      
      USE PATHS_thvarM
      USE PATHS_varcpld, ONLY: oneline, iffile, &
                               DOT, SLASH, BANG, BLANK, AST, &
                               cardname, mxncol, sc, &
                               fdum, coor, idum, paths_id 
      USE PATHS_util, ONLY: form7, nfields, toupper, read1more
      USE PATHS_allocM

      IMPLICIT NONE
      !
      ! ================================================================================
      ! WARNING: VARIABLE ASSIGNMENTS BETWEEN THE <defaults> TAGS ARE USED BY A PARSER
      !          TO PULL DEFAULT VALUES FROM THIS SUBROUTINE AND PLACE THEM IN THE
      !          PARCS INPUT MANUAL.  THE <options> TAGS ARE USED TO MAP CODE VARIABLES
      !          TO MAUNAL VARIABLES.  THE <defaults> AND <options> TAGS SHOULD NOT BE
      !          MODIFIED OR DELETED UNLESS YOU KNOW WHAT YOU ARE DOING.
      !
      ! <defaults cardgroup="Core Geometry" drop_prefix="PATHS_">
      ! ================================================================================
      !
      inpfile = ''
      outfile = ''
      fid_inp = 0
      fid_out = 0
      fid_pth = 0
      fid_xth = 0
      !
      !<options cardgroup=Initialization field_code="paths_id"/>
      paths_id = 'paths_default'
      !
      paths_onlyflag = .FALSE.
      !
      PATHS_fracdc=zero   ! Direct Heat Fraction
      PATHS_fracdvb=zero  ! Vessel Bypass Direct Heat Fraction
      PATHS_fracdwr=zero  ! Water Rod Direct Heat Fraction
      !
      freezetf=.FALSE.
      frozentf=26.85
      freezedc=.FALSE.
      frozendc=0.8
      !
      plevnew = 1.0
      !
      thsolver=1
      !
      ! Added by Linsen Li 10/26/2010
      PATHS_epsout=1.0e-6
      PATHS_epsin=1.0e-6
      PATHS_noutmax=100
      PATHS_ninmax=100000
      PATHS_thur=0.75
      PATHS_mscb=1 !subcooled quality correlation, default: EPRI
      PATHS_mvd=1  !void-quality correlation, default: EPRI
      PATHS_mfr=3  !wall drag correlation, default: Smooth laminar-turbulent correlation by Yunlin Xu  
      PATHS_mtpf=4 !two phase multiplier correlation, default: Martinelli-Nelson-Jones
      PATHS_rct=1
      PATHS_pfa=0.1524         !
      !
      PATHS_Tin=0.0
      PATHS_hin=0.0
      !
      !
      coor = 1  ! default to rectangular, hex/cly not supported in stand alone version
      !
      nfbxy = 0
      nrowy = 0 
      nchfb = 0
      nchby = 0
      nchwr = 0
      nzpl  = 0
      nzlr  = 0
      nzur  = 0
      !
      nchm = 0 
      !
      PATHS_nchan=0
      PATHS_nchantype=0
      PATHS_nr=10
! For verbosity control
      PATHS_verbos=0
! For printing extra debugging information      
      PATHS_debug=0
! For coupled print frequency
      PATHS_prntfq=100      
      !
      ! Reactangular sym
      isymmetry = 1
      ! Hexagonal angle
      isymang = 0.0
      ! Relaxation
      pow_rlx = 1.0
      
      ! User specified conductivities default off
      num_kfuel = 0
      num_kclad = 0
      
      ! Steam Tables
      stable = .FALSE.
      
      ! Steam Table additional option set to 0, no effect
      stm_opt = 0
      
      ! Track which cards are input for tests/checks
      card_info = 0      
      ! ================================================================================
      ! </defaults>
      ! ================================================================================

      ! 1st solve is initially TRUE
      first_solve = .TRUE.   
      !     
      ! ================================================================================
      ! WARNING: VARIABLE ASSIGNMENTS BETWEEN THE <defaults> TAGS ARE USED BY A PARSER
      !          TO PULL DEFAULT VALUES FROM THIS SUBROUTINE AND PLACE THEM IN THE
      !          PARCS INPUT MANUAL.  THE <options> TAGS ARE USED TO MAP CODE VARIABLES
      !          TO MAUNAL VARIABLES.  THE <defaults> AND <options> TAGS SHOULD NOT BE
      !          MODIFIED OR DELETED UNLESS YOU KNOW WHAT YOU ARE DOING.
      !
      ! <defaults cardgroup="Channel Geometry" drop_prefix="PATHS_">
      ! ================================================================================
      !
      PATHS_aopt=0  !No irreversible loss due to abrupt area change
! For chimney         
      PATHS_usechimney=.FALSE.
! For variable flow area
      PATHS_acard=.FALSE.
      !
      ! ================================================================================
      ! </defaults>
      ! ================================================================================
      !
      
END SUBROUTINE PATHS_default

