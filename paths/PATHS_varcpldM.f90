MODULE PATHS_varcpld

      USE IntrType, ONLY: sdk, sik
      
      IMPLICIT NONE
      
      ! renamed caseid
      CHARACTER(len=80):: paths_id 

      ! PARCS PowerM, plevel=1.0 is 100% power
      REAL(sdk):: plevnew

      ! PARCS PowerM, relp(:,:)=1.0 is 100% power, CALL dmalloc0(relp,0,nzth,0,nchan)
      REAL(sdk), POINTER, DIMENSION(:,:):: relpnew    !(0:nzth,0:nfbxy) ,relative power excludes chimney

      ! PARCS TimesM
      REAL(sdk):: tlap, tth
      
      ! PARCS, Fuelth
!      REAL(sdk):: akfuel(0:5)
!      REAL(sdk):: akclad(0:3)
!      REAL(sdk):: arcpfuel(0:3)
!      REAL(sdk):: arcpclad(0:3)     

      INTEGER(sik) :: mxncol, mxnfield
      PARAMETER (mxncol=1024,mxnfield=300)

      LOGICAL :: iffile
      
      ! Miscellaneous testing 
      CHARACTER(len=1):: DOT
      CHARACTER(len=1):: BANG
      CHARACTER(len=1):: BLANK
      CHARACTER(len=1):: SLASH
      CHARACTER(len=1):: AST      
      CHARACTER(len=1):: TAB
      PARAMETER (DOT='.',BANG='!',BLANK=' ',SLASH='/',AST='*',TAB=achar(9))
      
      ! Reading for input
      CHARACTER(len=mxncol) :: oneline
      CHARACTER(len=1)      :: sc(mxncol)
      
      INTEGER(sik), PARAMETER :: fid_scn=6
      INTEGER(sik) :: fid_inp, fid_out, fid_pth, fid_xth, fid_rlp

      ! File controls, debugging
      INTEGER(sik) :: fid_tmp1,fid_tmp2,fid_tmp3,fid_tmp4,fid_tmp5,fid_tmp6,fid_tmp7,fid_tmp8

      ! reading variable for card title name
      CHARACTER(len=10) :: cardname
      
      ! read variable for geometry
      INTEGER(sik) :: coor
      
      ! Scratch reading arrays for float/real and integer
      REAL(sdk):: fdum(mxnfield)
      INTEGER(sik):: idum(mxnfield)
      LOGICAL :: ldum(mxnfield)      

      ! Value 0.0
      REAL(sdk) :: zero
      PARAMETER (zero=0.0)
      
      ! Value of 273.15 conversion
      REAL(sdk) :: ckelvin
      PARAMETER (ckelvin=273.15)      

      ! To know if the exectiuon is the first
      LOGICAL :: first_solve          
      
      ! Rectangular symmetry option
      INTEGER(sik):: isymmetry 
      ! Hexagonal symmetry angle
      INTEGER(sik):: isymang

      ! Hex Geom variables
      INTEGER(sik):: icoreys            ! start y-coordinate for assembly, determined from rand_conf. when hexbyrow, it should
                                        ! be same as icordys for minimal inputs
      INTEGER(sik):: icoreye            ! end   y-coordinate for assembly, determined from rand_conf. when hexbyrow, it should
                                        ! be same as icordye for minimal inputs
                                        
      INTEGER(sik) :: num_cards
      PARAMETER (num_cards = 35)

      CHARACTER(len=10):: inp_card(num_cards)
      DATA inp_card &
           /'PATHS_ID  ','TH_IPATHS ','VERBOSITY ','COMM_PARM ','RELAX_POW ',  &
            'TH_LSOLVER','CONV_PARM ','TH_CORR   ','S_TABLE   ','FUEL_COND ',  &
            'CLAD_COND ','CPR_OPT   ','AREA_OPT  ','FREEZETF  ','FREEZEDC  ',  &
            'CHIMNEY   ','CORE_STATE','NK_RADIAL ','TH_RADIAL ','TH_AXIAL  ',  &
            'AX_MESH   ','TH_CONF   ','CH_TYPE   ','CH_T_MAP  ','CHAN_GEOM ',  &
            'KA_CHAN   ','AREA_CARD ','AREA_CHAN ','NROD_CHAN ','HD_CHAN   ',  &
            'AREA_LOSS ','ORIF_MAP  ','ORIF_LOSS ','TH_USEDEP ','TH_RELP   '/
      
      INTEGER(sik) :: card_info(num_cards)                                        
      
      INTEGER(sik), PARAMETER :: min_search_unit =  8  ! lowest allowable unit
      INTEGER(sik), PARAMETER :: max_search_unit = 99  ! max allowable unit                
      
 CONTAINS      
      
      SUBROUTINE get_blnk_unit(fid_value)
      ! This routine allows open unit values to be searched
      ! and then sends back the first open one
      ! It should prevent errors with NAG, or coupled
      
          IMPLICIT NONE
            
          INTEGER(sik), INTENT(INOUT) :: fid_value
          
          INTEGER(sik) :: io_flg, uu
          LOGICAL :: io_open
          
          DO uu = min_search_unit,max_search_unit 
              INQUIRE(unit = uu, OPENED=io_open, IOSTAT=io_flg)
              IF (io_flg .NE. 0) CYCLE
              IF (.NOT. io_open) EXIT
          END DO    
          
          fid_value = uu
          RETURN
          
      END SUBROUTINE get_blnk_unit       
      
END MODULE PATHS_varcpld
