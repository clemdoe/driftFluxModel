!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
SUBROUTINE PATHS_read(read_opt)
      ! --------------------------------------------------------------------------- !
      !           Parameters reading of PATHS for PARCS                     !
      ! --------------------------------------------------------------------------- !
      !
      ! Main part of subroutine --------------------------------------------------- !

      USE IntrType, ONLY: sdk, sik
      USE PATHS_thvarM
      USE PATHS_varcpld, ONLY: oneline, fid_inp, fid_out, &
                               DOT, SLASH, BANG, BLANK, AST, &
                               cardname, mxncol, sc, coor, &
                               fdum, idum, ldum, paths_id, &
                               get_blnk_unit
      USE PATHS_util, ONLY: form7, nfields, toupper, read1more
      USE PATHS_allocM

      IMPLICIT NONE

      INTEGER(sik), INTENT(IN) :: read_opt

      ! read iteration control data from PATHS_PARM block
      
      LOGICAL :: in_use

      INTEGER(sik) :: thconf(nfbxy)
      INTEGER(sik) :: i, j, k, indev,chan
      INTEGER(sik) :: ndataf, ndatafnew, ios
      INTEGER(sik) :: ja, ks, ke, io_flg
      INTEGER(sik) :: iPATHS_nchan,iPATHS_nchantype
      LOGICAL :: io_exist, io_open
      !
      INTEGER(sik) :: inp_test
      !
      CHARACTER(LEN=2) :: io_val
      CHARACTER(LEN=4) :: inp_opt
      CHARACTER(LEN=40) :: mesg
      CHARACTER(LEN=79) :: tmpfile      
      !
      indev=fid_inp      
      !      
      ndataf=0
      !
      IF (read_opt .EQ. 2) THEN
          PATHS_aloss = 0.0
          PATHS_kloss = 0.0
          PATHS_prel = 0.0
      END IF
      !
      SELECT CASE (read_opt)
      CASE(0)
          inp_opt = 'CHCK'
          IF (fid_out .EQ. 0) THEN
              CALL get_blnk_unit(fid_out)
              tmpfile = TRIM(inpfile)//'_paths_err'
              OPEN(fid_out,file=tmpfile, status='unknown')                
          ELSE    
              INQUIRE(fid_out,OPENED=in_use)
              IF (.NOT. in_use) THEN
                  CALL get_blnk_unit(fid_out)
                  tmpfile = TRIM(inpfile)//'_paths_err'
                  OPEN(fid_out,file=tmpfile, status='unknown')                
              END IF
          END IF      
      CASE(1)
          inp_opt = 'SCAN'
      CASE(2)
          inp_opt = 'READ'
          pwr_inp = 0
      END SELECT
      !
      rloop: DO WHILE(.TRUE.)
!         DO WHILE (.TRUE.)
            READ(indev,'(a1024)',IOSTAT=ios) oneline
            IF (ios .GT. 0 .OR. ios .LT. -1) THEN
                WRITE(io_val,'(I2)') ios
                mesg = 'Error during read. IOSTAT given as value: '//io_val
                CALL input_stp(mesg)
            END IF
            IF (ios .EQ. -1) THEN
                mesg='Error during read. End of file reached. Check for final .(dot)'
                CALL input_stp(mesg)
            END IF            
            ! linsen: DOT='.',BANG='!',BLANK=' ',SLASH='/',AST='*'
            IF(oneline(1:1).EQ.BANG .OR. oneline.EQ.BLANK) CYCLE
            ! end of input exit or stacked case in legacy input
            IF(oneline(1:1).EQ.DOT) THEN
                EXIT rloop
            END IF
            !
            SELECT CASE (read_opt)
            CASE(0)
               IF(oneline(1:1) .EQ. TAB) CYCLE
            CASE(1)
               IF(oneline(1:1) .EQ. TAB) THEN
                   mesg='Error: Remove tabs from input'
                   CALL input_err(mesg,cardname)                     
               END IF
            CASE(2)
            END SELECT
            !
            ! linsen: FMT=*: List-Directed Data Transfer
            IF(oneline.NE.BLANK) READ(oneline,*,IOSTAT=ios) cardname
            ! linsen: convert lower cardname to upper cardname
            CALL toupper(cardname)
            !
            ndataf=nfields(oneline)-1
            !
            SELECT CASE(cardname)
               ! ----------------------------------------------------------------------
            CASE('PATHS_ID')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="PATHS_ID" description="Assignment of name for output"
            ! "from \ac{PATHS} calculation.">
            ! \Card{PATHS_ID}
            !
            ! This card is used to assign a name for the output from the \ac{PATHS} calculation.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{paths_id}{String}{Default = \default}
            ! Alphanumeric case ID for the \ac{PATHS} calculation used to name the output files.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below assigns the filename PATHS_JOB to the \ac{PATHS} output files.
            !
            ! \begin{examplebox}
            ! !           paths_id
            ! PATHS_ID    PATHS_JOB
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(1) = card_info(1) + 1
                    READ(oneline,*,IOSTAT=ios) cardname,paths_id
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 1) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                        
                    END IF                               
                CASE DEFAULT
                END SELECT
            CASE('TH_IPATHS')  ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="TH_IPATHS" description="Selects between the "
            ! "standalone/coupled \ac{T/H} calculation.">
            ! \Card{TH_IPATHS}
            !
            ! This card is used to select between the standalone or coupled \ac{PATHS} \ac{T/H} calculation.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{paths_onlyflag}{Boolean}{Default = N/A}
            ! \ac{PATHS} \ac{T/H} calculation flag.
            ! Set void fraction quality correlation:
			!
            ! \begin{fldoptions}
            !    \fldopt{T} \ac{PATHS} \ac{T/H} calculation is performed standalone, without neutronics.
            !    \fldopt{F} \ac{PATHS} calculation is copuled to PARCS neutronics results
            ! \end{fldoptions}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !            paths_onlyflag
            ! TH_IPATHS          F
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(2) = card_info(2) + 1
                    IF (ndataf .EQ. 1) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,paths_onlyflag
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE IF (ndataf .EQ. 2) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,paths_onlyflag,paths_prntfq
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE 
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                        
                    END IF
                CASE DEFAULT
                END SELECT
            CASE('VERBOSITY')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="VERBOSITY" description="Activates additional data for"
            ! "debugging or output from the calculation (Developer Option).">
            ! \Card{VERBOSITY}
            !
            ! This card is used to select how much additional data is provided in the \ac{PATHS} output,
            ! and if debugging data will be included. If \fldlink{PATHS_verbos} is 0, no
            ! additional output data will be provided. A value of 1 includes the additional data
            ! from the calculation in the output.
            !
            ! If \fldlink{PATHS_debug} is 0, no debugging data is printed to the output. A value
            ! of 1 includes the additional debugging information.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{PATHS_verbos}{Integer}{Default = 0}
            ! \ac{PATHS} verbose data output flag.
            !
            ! \fldhdr{PATHS_debug}{Integer}{Default = 0}
            ! \ac{PATHS} debug data output flag.
            !
            ! \warning{This card is primarily for use by developers - it is not recommended for
            !   most users.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the VERBOSITY card with the default values set.
            ! \begin{examplebox}
            ! !            PATHS_verbos      PATHS_debug
            ! VERBOSITY          0                0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(3) = card_info(3) + 1
                CASE(1)
                    IF (ndataf .EQ. 1) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,PATHS_verbos
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE IF (ndataf .EQ. 2) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,PATHS_verbos,PATHS_debug
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF   
                    IF (PATHS_verbos .LT. 0 .OR. PATHS_verbos .GT. 2) THEN
                        mesg='verbosity is outside of range [0:2]'
                        CALL input_err(mesg,cardname)
                    END IF                    
                    IF (PATHS_debug .LT. 0 .OR. PATHS_debug .GT. 2) THEN
                        mesg='debug is outside of range [0:2]'
                        CALL input_err(mesg,cardname)
                    END IF                     
                CASE DEFAULT
                END SELECT                
            CASE('COMM_PARM') ! required if coupled
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="COMM_PARM" description="The COMM_PARM card sets"
            ! "global core values.">
            ! \Card{COMM_PARM}
            !
            ! The input fields on this card set the number of discrete rings in the fuel and
            ! the fuel assembly pitch.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{nr}{Integer}{Default = \default}
            ! Number of discrete rings in the fuel.
            !
            ! \fldhdr{pfa}{Real}{Default = \default}
            ! Assembly pitch (m).
            !
            ! \note{The COMM_PARM card is used only for the PARCS neutronics initialization.
            ! for standalone \ac{PATHS} calculations, the COMM_PARM card may be omitted.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the COMM_PARM card with the default values.
            !
            ! \begin{examplebox}
            ! !              nr      pfa
            ! COMM_PARM      10     0.1524
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(4) = card_info(4) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,PATHS_nr,PATHS_pfa
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 2) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                        
                    END IF                      
                    IF (PATHS_nr .LT. 2 .OR. PATHS_nr .GT. 25) THEN
                        mesg='Nrings for conduction solve is out of range'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_pfa .LT. 0.0 .OR. PATHS_pfa .GT. 1.0) THEN
                        mesg='PATHS_pfa for conduction solve is out of range [0.0:1.0]'
                        CALL input_err(mesg,cardname)
                    END IF 
                CASE DEFAULT
                END SELECT
            CASE('RELAX_POW') ! optional if coupled
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="RELAX_POW" description="Specify the relaxation parameter"
            ! "for the power coupling between \ac{PATHS} and PARCS.">
            ! \Card{RELAX_POW}
            !
            ! A value of 1.0 for the relaxation parameter between the \ac{PATHS} and PARCS calculation
            ! will use the latest power values from PARCS in the \ac{PATHS} calculation, while any
            ! value less than 1.0 will include a weighting with the power values from the
            ! previous iteration. A value of less than 1.0 will likely slow convergence in 
            ! well-behaved problems, but it may improve convergence in some coupled (``stiff'')
            ! problems.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{pow_rlx}{Real}{Default = 1.0}
            ! Power relaxation parameter between PARCS and \ac{PATHS}.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the RELAX_POW card with the default value set.
            !
            ! \begin{examplebox}
            ! !            pow_rlx
            ! RELAX_POW      1.0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(5) = card_info(5) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname, pow_rlx
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 1) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                        
                    END IF                      
                    IF (pow_rlx .LT. 1e-4 .OR. pow_rlx .GT. 1) THEN
                        mesg='pow_rlx is out of range, please revise input'
                        CALL input_err(mesg,cardname)
                    END IF                    
                CASE DEFAULT
                END SELECT                
            CASE('TH_LSOLVER')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="TH_LSOLVER" description="Selection of the linear solver"
            ! "used in \ac{PATHS}.">
            ! \Card{TH_LSOLVER}
            !
            ! This card is used to select the linear solver used for the solution of the combined
            ! mass/momentum system of equations.
            !
            ! \note{This card may be omitted because only one solver (\ac{BiCGSTAB}) is currently available
            ! in the code.}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{thsolver}{Integer}{Default = 1}
            ! Solver for mass/momentum system of equations.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the TH_LSOLVER card with the default values set.
            !
            ! \begin{examplebox}
            ! !            thsolver
            ! TH_LSOLVER      1
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(6) = card_info(6) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,thsolver
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 1) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                        
                    END IF                    
                    IF (thsolver .LT. 1 .OR. thsolver .GT. 2) THEN
                        mesg='Invalid th solver option'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (thsolver .EQ. 2) THEN
                        mesg='Error: Gause Seidel Solver under development'
                        CALL input_err(mesg,cardname)
                    END IF                    
                CASE DEFAULT
                END SELECT
            CASE('CONV_PARM')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="CONV_PARM" description="Sets various values that"
            ! "control the calculation.">
            ! \Card{CONV_PARM}
            !
            ! This card is used to assign the maximum number of inner and outer iterations and
            ! the convergence criteira for the inner and outer iterations.
            !
            ! The input for the outer iterations applies to the non-linear \ac{PATHS} physics solution.
            ! The inner iteration input is for the Newton-Krylov solver.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{noutmax}{Integer}{Default = 100}
            ! Maximum number of outer iterations.
            !
            ! \fldhdr{epsout}{Real}{Default = 1.0E-06}
            ! Convergence criterion for outer iterations.
            !
            ! \fldhdr{ninmax}{Integer}{Default = 1E+05}
            ! Maximum number of inner iterations.
            !
            ! \fldhdr{epsin}{Real}{Default = 1.0E-06}
            ! Convergence criterion for inner iterations.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the CONV_PARM card with the default values set.
            !
            ! \begin{examplebox}
            ! !            noutmax   epsout     ninmax      epsin
            ! CONV_PARM      100     1.E-06     1E+05       1.E-06
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(7) = card_info(7) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,(fdum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .LT. 4 .OR. ndataf .GT. 5) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF                 
                    IF (ndataf .EQ. 5) WRITE(fid_out,*)'Warning | 5th entry in ',cardname,' is currently ignored'
                    PATHS_noutmax=fdum(1)
                    PATHS_epsout=fdum(2)
                    PATHS_ninmax=fdum(3)
                    PATHS_epsin=fdum(4)
                    ! This input was being misused, so disabled for now.
                    ! IF(ndataf.GT.4) PATHS_thur=fdum(5)
                    IF (PATHS_noutmax .LT. 1 .OR. PATHS_noutmax .GT. 1e10) THEN
                        mesg='PATHS_noutmax is out of range, please revise input'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_epsout .LT. 1e-16 .OR. PATHS_epsout .GT. 1) THEN
                        mesg='PATHS_epsout is out of range, please revise input'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_ninmax .LT. 1 .OR. PATHS_ninmax .GT. 1e10) THEN
                        mesg='PATHS_ninmax is out of range, please revise input'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_epsout .LT. 1e-16 .OR. PATHS_epsout .GT. 1) THEN
                        mesg='PATHS_epsin is out of range, please revise input'
                        CALL input_err(mesg,cardname)
                    END IF                    
                CASE DEFAULT
                END SELECT
            CASE('TH_CORR')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="TH_CORR" description="Set quality correlations and wall"
            ! "friction factors.">
            ! \Card{TH_CORR}
            !
            ! This input determines if the coolant density will be a constant or vary normally during the
            ! calculation.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{mvd}{Integer}{Default = \default}
            ! Set void fraction quality correlation:
			!
            ! \begin{fldoptions}
            !    \fldopt{1} \ac{EPRI}
            !    \fldopt{2} GE-Ramp
            !    \fldopt{3} Modified Bestion
            !    \fldopt{4} Homogeneous \ac{HEM}
            !    \fldopt{5} Alternate Modified Bestion model with C0=1.0
            !    \fldopt{6} Kataoka-Ishii
            !%    \fldopt{7} Liao, Parlos, Griffith - proprietary model, shouldn't be included in the manual.
            ! \end{fldoptions}
            !
            ! \fldhdr{mscb}{Integer}{Default = \default}
            ! Subcooled quality correlation:
			!
            ! \begin{fldoptions}
            !    \fldopt{1} \ac{EPRI}
            !    \fldopt{2} Equilibrium
            ! \end{fldoptions}
            !
            ! \fldhdr{mfr}{Integer}{Default = \default}
            ! Wall friction factor:
			!
            ! \begin{fldoptions}
            !    \fldopt{1} Blasius
            !    \fldopt{2} Churchill
            !    \fldopt{3} Smooth laminar-turbulent correlation by Yunlin Xu
            ! \end{fldoptions}
            !
            ! \fldhdr{mtpf}{Integer}{Default = \default}
            ! Two-phase multiplier for wall friction:
			!
            ! \begin{fldoptions}
            !    \fldopt{1} \ac{HEM} 1
            !    \fldopt{2} \ac{HEM} 2
            !    \fldopt{3} Martinelli-Nelson
            !    \fldopt{4} Martinelli-Nelson-Jones
            ! \end{fldoptions}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !               mvd        mscb    mfr    mtpf
            ! TH_CORR          1          1       3      4
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(8) = card_info(8) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,(idum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .LT. 1 .OR. ndataf .GT. 4) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF  
                    PATHS_mvd=idum(1)             ! void fraction-quality correlation:
                    IF(ndataf>=2)THEN
                       PATHS_mscb=idum(2)         ! subcooled quality correlation
                       IF(ndataf>=3)THEN
                          PATHS_mfr=idum(3)       ! wall friction correlation
                          IF(ndataf>=4)THEN
                             PATHS_mtpf=idum(4)   ! two phase multiplier for wall friction
                          ENDIF
                       ENDIF
                    ENDIF
                    IF (PATHS_mvd .LT. 1 .OR. PATHS_mvd .GT. 7) THEN
                        mesg='MVD value is outside of range [1:7]'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_mscb .LT. 1 .OR. PATHS_mscb .GT. 2) THEN
                        mesg='MSCB value is outside of range [1:2]'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_mfr .LT. 1 .OR. PATHS_mfr .GT. 3) THEN
                        mesg='MFR value is outside of range [1:3]'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_mtpf .LT. 1 .OR. PATHS_mtpf .GT. 4) THEN
                        mesg='MTPF value is outside of range [1:4]'
                        CALL input_err(mesg,cardname)
                    END IF                    
                CASE DEFAULT
                END SELECT                
            CASE('S_TABLE')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="S_TABLE" description="The S_TABLE card sets"
            ! "steam table and water properites options.">
            ! \Card{S_TABLE}
            !
            ! The \fldlink{stable} field indicates if a steam table or the built-in polynomial
            ! fits are to be used for water properites in the calculation. The \fldlink{stm_opt}
            ! field provides additional options for the water properties used.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{stable}{Boolean}{Default = \default}
            ! Steam table options:
			!
            ! \begin{fldoptions}
            !    \fldopt{T} Use built-in IAPWS steam tables.
            !    \fldopt{F} Use built-in polynomial fits for saturated fluid properties.
            ! \end{fldoptions}
            !
            ! \fldhdr{stm_opt}{Integer}{Default = \default}
            ! Additional option for water properties (optional input):
			!
            ! \begin{fldoptions}
            !    \fldopt{-1} Subcooling fix-up for internal polynomials (requires \fldlink{stable} = F).
            !    \fldopt{0} No effect, uses standard IAPWS data \cite{IAPWS97}.
            !    \fldopt{1} Reads external tpfh2onew water file data instead of the built-in IAPWS tables.
            ! \end{fldoptions}
            !
            ! \note{The tpfh2onew file must be in the same directory as the input file at execution time
            !    when stm_opt = 1.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !            stable    stm_opt
            ! S_TABLE        F          0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(9) = card_info(9) + 1                    
                CASE(1)
                    IF (ndataf .EQ. 1) THEN
                        READ(oneline,*,IOSTAT=ios) cardname, stable
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE IF (ndataf .EQ. 2) THEN
                        READ(oneline,*,IOSTAT=ios) cardname, stable, stm_opt
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                        
                    END IF                               
                    IF (stable .AND. stm_opt .LT. 0) THEN
                        mesg='Steam Tables & Subcooling Fix-up are not compatible, please revise input'
                        CALL input_err(mesg,cardname)
                    END IF
                CASE DEFAULT
                END SELECT                
            CASE('FUEL_COND')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="FUEL_COND" description="Specify a polynomial for fuel"
            ! "conductivity.">
            ! \Card{FUEL_COND}
            !
            ! This input allows a user to specify the polynomial for the fuel conductivity. This input
            ! provides the coefficients from the 0th order to the nth order.
            !
            ! The default polynomial is:
            ! \begin{equation}
            ! k_{fuel}\left(T\right)=2.3595\times 10^{-12}T^{4}-1.5236\times 10^{-8}T^{3}+3.6285\times 10^{-5}T^{2}-0.038352 T + 17.94
            ! \end{equation}
			!
			! where $k_{fuel}(T)$ is the fuel conductivity with units of $W/m\cdot{K}$ and $T$ is the temperature with units of $K$.
			!
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{dat_kfuel[num_kfuel]}{Real}{Default = See above}
            ! \ac{PATHS} will count the number of fields on the FUEL_COND card to determine the value of num_kfuel.
            !
            ! \note{The order of the polynomial is expected to be between 0 and 8.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example block below shows the correct FUEL_COND input consistent with the default
            ! polynomial shown above.
            !
            ! \begin{examplebox}[long]
            ! !               0th        1st         2nd            3rd            4th
            ! FUEL_COND      17.94   -3.8352E-02   3.6285E-05   -1.5236E-08     2.3595E-12
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(10) = card_info(10) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname, (fdum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .GT. 0) num_kfuel = ndataf
                    IF (num_kfuel .LT. 1 .OR. num_kfuel .GT. 9) THEN                    
                        mesg='Fuel thermal conductivity between 0th and 8th order is expected'
                        CALL input_err(mesg,cardname)
                    END IF
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname, (fdum(i),i=1,ndataf)
                    dat_kfuel = fdum
                CASE DEFAULT
                END SELECT
            CASE('CLAD_COND')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="CLAD_COND" description="Specify a polynomial for cladding"
            ! "conductivity.">
            ! \Card{CLAD_COND}
            !
            ! This input allows a user to specify the polynomial for the cladding conductivity. This input
            ! provides the coefficients from the 0th order to the nth order.
            !
            ! The default polynomial is:
            ! \begin{equation}
            ! k_{clad}\left(T\right)=3.0024\times 10^{-12}T^{4}-1.027\times 10^{-8}T^{3}+1.8274\times 10^{-5}T^{2}-0.004965 T + 11.839
            ! \end{equation}
            !
			! where $k_{clad}(T)$ is the cladding conductivity with units of $W/m\cdot{K}$ and $T$ is the temperature in units of $K$.
			!
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{dat_kcond[num_kcond]}{Real}{Default = See above}
            ! \ac{PATHS} will count the number of fields on the CLAD_COND card to determine the value of num_kcond.
            !
            ! \note{The order of the polynomial is expected to be between 0 and 8.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            ! The example block below shows the correct CLAD_COND input consistent with the default
            ! polynomial shown above.
            !
            ! \begin{examplebox}[long]
            ! !               0th        1st         2nd           3rd          4th
            ! CLAD_COND      11.839   -4.965E-03   1.8274E-05  -1.027E-08   3.0024E-12
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(11) = card_info(11) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname, (fdum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .GT. 0) num_kclad = ndataf
                    IF (num_kclad .LT. 1 .OR. num_kclad .GT. 9) THEN                    
                        mesg='Clad thermal conductivity between 0th and 8th order is expected'
                        CALL input_err(mesg,cardname)
                    END IF
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname, (fdum(i),i=1,ndataf)
                    dat_kclad = fdum
                CASE DEFAULT
                END SELECT  
            CASE('CPR_OPT')
               ! This option is purposefully not documented in the manual, contains controlled data
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(12) = card_info(12) + 1                    
                CASE(2)
                    IF (ndataf .NE. 1) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                    READ(oneline,*,IOSTAT=ios) cardname,PATHS_cpropt
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF(PATHS_cpropt .LT. 0 .OR. PATHS_cpropt .GT. 2) THEN
                       mesg='value outside of range [0:2]'
                       CALL input_err(mesg,cardname)
                    ENDIF
                CASE DEFAULT
                END SELECT  
            CASE('AREA_OPT')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="AREA_OPT" description="Option to include irreversible"
            ! "flow losses.">
            ! \Card{AREA_OPT}
            !
            ! The internal calculation for pressure change due to abrupt flow area changes includes
            ! reversible losses and user-input loss factors. This card activates the irreversible losses
            ! in the pressure change calculation.
            !
            !% \todo[inline]{There are some equations that show how the pressure drops are computed for
            !% reversible and irreversible conditions in the manual. They probably don't belong in this
            !% document, but should be included in the theory manual. GROUP=Channel Geometry Cards CARD=\CardName}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{aopt}{Integer}{Default = \default}
            ! Abrupt area change option:
			!
            ! \begin{fldoptions}
            !    \fldopt{0} Abrupt area change does not include irreversible losses.
            !    \fldopt{1} Irreversible losses are included in the pressure change calculation.
            ! \end{fldoptions}
            !
            ! \note{The internal correlations match the ones used in the TRACE code.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !               aopt
            ! AREA_OPT         0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(13) = card_info(13) + 1                    
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname,PATHS_aopt
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 1) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                    
                    IF (PATHS_aopt .NE. 0 .AND. PATHS_aopt .NE. 1) THEN
                         mesg='input must be 0 or 1. -1 not implemented'
                         CALL input_err(mesg,cardname)                          
                    ENDIF
                CASE DEFAULT
                END SELECT               
            CASE('FREEZETF')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="FREEZETF" description="Specify how fuel temperature will"
            ! "be computed.">
            ! \Card{FREEZETF}
            !
            ! This input determines if the fuel temperature will be a constant or vary normally during the
            ! calculation.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{freezetf}{Boolean}{Default = \default}
            !
            ! Fuel temperature variation option.
            !
            ! \begin{fldoptions}
            !    \fldopt{T} Make the fuel temperature value a constant.
            !    \fldopt{F} Allow the fuel temperature to vary normally.
            ! \end{fldoptions}
            !
            ! \fldhdr{frozentf}{Real}{Default = \default}
            ! Value of (constant) fuel temperature in Celsius. This input field must have a value
            ! between 1 and 3000.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !             freezetf  frozentf
            ! FREEZETF         T        800
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(14) = card_info(14) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,freezetf
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .GT. 2) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF                    
                    IF(ndataf.EQ.2 .AND. freezetf) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,freezetf,frozentf
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                        IF (frozentf .LT. 1.0 .OR. frozentf .GT. 3000.0) THEN
                            mesg='frozentf value is outside of range [1:3000]'
                            CALL input_err(mesg,cardname)
                        END IF                        
                    ELSE
                        WRITE(fid_out,*)cardname,' defaulting to 300K'
                    ENDIF
                    frozentf=SQRT(frozentf+273.15)                    
                CASE DEFAULT
                END SELECT
            CASE('FREEZEDC')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="FREEZEDC" description="Specify how coolant density will"
            ! "be computed.">
            ! \Card{FREEZEDC}
            !
            ! This input determines if the coolant density will be a constant or vary normally during the
            ! calculation.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{freezedc}{Boolean}{Default = \default}
            !
            ! Coolant density calculation option:
            !
            ! \begin{fldoptions}
            !    \fldopt{T} Make the coolant density value a constant.
            !    \fldopt{F} Allow the coolant density to vary normally.
            ! \end{fldoptions}
            !
            ! \fldhdr{frozendc}{Real}{Default = \default}
            ! Value of (constant) coolant density in g/cc. This input should have a value between 0.001 and 1.0.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !             freezedc  frozendc
            ! FREEZEDC         T        0.71
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(15) = card_info(15) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,freezedc
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 2) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF                    
                    IF(ndataf.EQ.2 .AND. freezedc) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,freezedc,frozendc
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                        IF (frozendc .LT. 0.001 .OR. frozendc .GT. 1.0) THEN
                            mesg='frozendc value is outside of range [0.001:1]'
                            CALL input_err(mesg,cardname)
                        END IF                        
                    ELSE
                        WRITE(fid_out,*)cardname,' defaulting to 0.8 g/cm^3'
                    ENDIF
                    frozendc=frozendc*1000.0           
                CASE DEFAULT
                END SELECT
            CASE('CHIMNEY')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="CHIMNEY" description="Add unheated nodes to the top"
            ! "and bottom of the channels.">
            ! \Card{CHIMNEY}
            !
            ! This input sets if an unheated node is to be added to the top of each T/H channel,
            ! and if so, how long it should be.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{usechimney}{Boolean}{Default = \default}
            !
            ! Chimney option that adds unheated nodes to \ac{PATHS} calculation.
            !
            ! \begin{fldoptions}
            !    \fldopt{T} Add an extra unheated node (seen only by \ac{PATHS}) to the top of each
            !    T/H channel.
            !    \fldopt{F} Do not add unheated nodes to the top of the channels.
            ! \end{fldoptions}
            !
            ! \fldhdr{hzchimney}{Real}{Default = N/A}
            ! If \fldlink{usechimney} = T, input the height of the chimney region in cm.
            !
            ! \note{Users can now specify unique axial nodalizations between \ac{PATHS} and PARCS, so
            ! the CHIMNEY card is no longer necessary for adding an unheated node to \ac{PATHS}.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !            usechimney  hzchimney
            ! CHIMNEY          T        100.0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(16) = card_info(16) + 1                      
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,PATHS_usechimney,PATHS_hzchimney
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 2) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF  
                    IF (PATHS_hzchimney .LT. 0.1 .OR. PATHS_hzchimney .GT. 500.0) THEN
                        mesg='hzchimney is outside of range [0.1:500]'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_usechimney) nchm = 1
                CASE DEFAULT
                END SELECT                
            CASE('CORE_STATE') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Initialization" card="CORE_STATE" description="Sets the initial core"
            ! "conditions for the \ac{PATHS} calculation.">
            ! \Card{CORE_STATE}
            !
            ! This is a required card that must be provided in every input file. It is used
            ! to assign the initial values of total thermal power, flow rate, inlet enthalpy
            ! and outlet pressure for the core.
            !
            ! \note{If the STATE_CORE card is provided in the PARCS input, the information on this
            !       card will be overwritten, and the data from the STATE_CORE card in PARCS will be
            !       used \cite{PARCSUserV2}.}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{rtp}{Real}{Default = N/A}
            ! Total thermal power to all channels ($kW$).
            !
            ! \fldhdr{mdotcore}{Real}{Default = N/A}
            ! Total active flow rate to all channels ($kg/s$).
            !
            ! \note{There is currently no bypass model implemented, so mdotcore should only
            ! include only active flow, not bypass flow nor water rod flow.}
            !
            ! \note{rtp and mdotcore should be given as full-core values, even when using a core symmetry option in PARCS.}
            !
            ! \fldhdr{hin}{Real}{Default = N/A}
            ! If hin > 0, hin is the core inlet enthalpy ($kJ/kg$). If hin < 0, then hin is the
            ! core inlet temperature ($K$).
            !
            ! Only input a negative value of hin if \fldlink[S_TABLE]{stable}=TRUE (see \cardref[Core Geometry]{S_TABLE}).
            ! This is because the steam table subroutine is the only current option available to convert
            ! the inlet temperature into inlet enthalpy internally.
            !
            ! \fldhdr{Pout}{Real}{Default = N/A}
            ! Core outlet pressure ($Pa$). This is the pressure applied at the top edge of each channel.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below assigns the thermal power, total flow, inlet enthalpy, and outlet pressure
            ! to be 2949.0 $kW$, 9.506 $kg/s$, 1212.8 $kJ/kg$, and 7.18E6 $Pa$, respectively.
            !
            ! \begin{examplebox}
            ! !              RTP     mdotcore    hin         Pout
            ! CORE_STATE    2949.0    9.506     1212.8     7.18E+06
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(17) = card_info(17) + 1                    
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,(fdum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 4) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF                     
                    PATHS_RTP=fdum(1)
                    PATHS_mdotcore=fdum(2)
                    IF (fdum(3) .LT. 0) THEN
                        PATHS_Tin= -1.0 * fdum(3)
                    ELSE
                        PATHS_hin=fdum(3)
                    END IF
                    PATHS_Pout=fdum(4)
                    IF (PATHS_RTP .LT. 0.0 .OR. PATHS_RTP .GT. 1e10) THEN
                        mesg='power is outside of range [0:1e10]'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (PATHS_mdotcore .LT. 1.0 .OR. PATHS_mdotcore .GT. 1e6) THEN
                        mesg='core flow is outside of range [1:1e6]'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (DABS(fdum(3)) .LT. 0.1 .OR. DABS(fdum(3)) .GT. 1e4) THEN
                        mesg='inlet temp/enth is outside of range [0.1:1e4]'
                        CALL input_err(mesg,cardname)
                    END IF                    
                    IF (PATHS_Pout .LT. 1e2 .OR. PATHS_Pout .GT. 1e8) THEN
                        mesg='outlet pressure is outside of range [1e2:1e8]'
                        CALL input_err(mesg,cardname)
                    END IF                    
                CASE DEFAULT
                END SELECT
            CASE('NK_RADIAL') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="NK_RADIAL" description="The NK_RADIAL card sets"
            ! "geometric parameters and radial properties of the core.">
            ! \Card{NK_RADIAL}
            !
            ! The \fldlink{coor} field on this card is used to select between rectangular
            ! and hexagonal coordinate systems for core input. The \fldlink{nfbxy} and
            ! \fldlink{nrowy} fields set the number of fuel assemblies in the core and the
            ! number of rows used in data input, respectively.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{coor}{Integer}{Default = \default}
            !
            ! Coordinate system selection:
            !
            ! \begin{fldoptions}
            !    \fldopt{1} Core geometry will be defined using a rectangular coordinate system.
            !    \fldopt{3} Core geometry uses a hexagonal coordinate system.
            ! \end{fldoptions}
            !
            ! \fldhdr{nfbxy}{Integer}{Default = \default}
            ! Number of fuel assemblies in the core.
            !
            ! \fldhdr{nrowy}{Integer}{Default = \default}
            ! Number of rows in the fuel assembly layout on cards \cardlink[Channel Geometry]{TH_CONF},
            ! \cardlink[Channel Geometry]{CH_T_MAP} and \cardref[Channel Geometry]{ORIF_MAP}.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the NK_RADIAL card indicating a rectangular geometry in a
            ! core with 820 fuel assemblies and 34 rows in the fuel assembly input.
            !
            ! \begin{examplebox}
            ! !                coor      nfbxy     nrowy
            ! NK_RADIAL          1        820       34
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(18) = card_info(18) + 1                      
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,(idum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    coor = idum(1)
                    nfbxy = idum(2)
                    nrowy = idum(3)
                    IF (ndataf .LT. 3 .OR. ndataf .GT. 4) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF  
                    IF (coor .NE. 1 .AND. coor .NE. 3) THEN
                        mesg='coordinate is outside of range [1,3]'
                        CALL input_err(mesg,cardname)
                    END IF       
                    IF (nfbxy .LT. 1 .OR. nfbxy .GT. 1e4) THEN
                        mesg='nfbxy is outside of range [1:1e4]'
                        CALL input_err(mesg,cardname)
                    END IF  
                    IF (nrowy .LT. 1 .OR. nrowy .GT. 100) THEN
                        mesg='nrowy is outside of range [1:100]'
                        CALL input_err(mesg,cardname)
                    END IF                     
                    IF (coor .EQ. 3) THEN ! If Hex, extra data needed
                        IF (ndataf .NE. 4) THEN
                            mesg='Please provide the desired hex symmetry angle'
                            CALL input_err(mesg,cardname)
                        ELSE
                            isymang = idum(4)
                            IF (MOD(isymang,30) .NE. 0) THEN
                               mesg='Symmetry angle must be a multiple of 30, less than 360'
                               CALL input_err(mesg,cardname)
                            END IF
                        END IF
                        ALLOCATE(hcols(nrowy,2))
                    END IF
                CASE DEFAULT
                END SELECT
            CASE('TH_RADIAL') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="TH_RADIAL" description="The TH_RADIAL card sets"
            ! "radial parameters for the channels in the core.">
            ! \Card{TH_RADIAL}
            !
            ! The input fields on this card indicate the number of \ac{T/H} channels, the number of
            ! bypass channels, and the number of water rod channels in the core.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{nchfb}{Integer}{Default = \default}
            ! Number of \ac{T/H} channels. This value must be 1 or larger, and less than or equal to
            ! \fldlink[Core Geometry,NK_RADIAL]{nfbxy} (1<=nchfb<=\fldlink[Core Geometry,NK_RADIAL]{nfbxy}).
            ! When nchfb < \fldlink[Core Geometry,NK_RADIAL]{nfbxy}, some fuel assemblies share a
            ! channel - meaning that a single channel can contain multiple fuel assemblies.
            !
            ! \note{This will be clarified in future \ac{PATHS} releases.}
            !
            ! \fldhdr{nchby}{Integer}{Default = \default}
            ! Number of bypass channels.
            !
            ! \note{This input is currently unused, but is reserved for future use.}
            !
            ! \fldhdr{nchwr}{Integer}{Default = \default}
            ! Number of water rod channels.
            !
            ! \note{This input is currently unused, but is reserved for future use.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the TH_RADIAL card indicating 600 \ac{T/H} channels in a core.
            !
            ! \begin{examplebox}
            ! !             nchfb     nchby     nchwr
            ! TH_RADIAL     600       0         0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(19) = card_info(19) + 1                      
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,(idum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 3) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF                     
                    nchfb = idum(1)
                    nchby = idum(2)
                    nchwr = idum(3)
                    IF (nchfb .LT. 1 .OR. nchfb .GT. 1e4) THEN
                        mesg='nchfb is outside of range [1:1e4]'
                        CALL input_err(mesg,cardname)
                    END IF 
                    IF (nchby .NE. 0) THEN
                        mesg='zero nchby is the only allowed value at this time'
                        CALL input_err(mesg,cardname)
                    END IF 
                    IF (nchwr .NE. 0) THEN
                        mesg='zero nchwr is the only allowed value at this time'
                        CALL input_err(mesg,cardname)
                    END IF                    
                CASE DEFAULT
                END SELECT
            CASE('TH_AXIAL') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="TH_AXIAL" description="The TH_AXIAL card sets"
            ! "the number of heated and unheated axial nodes in the \ac{T/H} channels.">
            ! \Card{TH_AXIAL}
            !
            ! Three input fields on this card set the total number of axial nodes, and the numbers
            ! of unheated nodes in the lower and upper reflector regions.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{nzpl}{Integer}{Default = \default}
            ! Total number of axial nodes (heated plus unheated).
            !
            ! \fldhdr{nzlr}{Integer}{Default = \default}
            ! Number of unheated nodes in the lower reflector region.
            !
            ! \fldhdr{nzur}{Integer}{Default = \default}
            ! Number of unheated nodes in the upper reflector region.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the TH_AXIAL card indicating 27 total axial nodes with 1
            ! unheated lower reflector node and 1 unheated upper reflector node.
            !
            ! \begin{examplebox}
            ! !              nzpl       nzlr      nzur
            ! TH_AXIAL        27         1         1
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(20) = card_info(20) + 1                      
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,(idum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 3) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF                      
                    nzpl = idum(1)
                    nzlr = idum(2)
                    nzur = idum(3)
                    IF (nzpl .LT. 1 .OR. nzpl .GT. 1000) THEN
                        mesg='nzpl is outside of range [1:1000]'
                        CALL input_err(mesg,cardname)
                    END IF 
                    IF (nzlr .LT. 0 .OR. nzlr .GT. 20) THEN
                        mesg='nzlr is outside of range [0:20]'
                        CALL input_err(mesg,cardname)
                    END IF 
                    IF (nzur .LT. 0 .OR. nzur .GT. 20) THEN
                        mesg='nzur is outside of range [0:20]'
                        CALL input_err(mesg,cardname)
                    END IF 
                CASE DEFAULT
                END SELECT
            CASE('AX_MESH') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Geometry" card="AX_MESH" description="Sets"
            ! "axial node size in the core.">
            ! \Card{AX_MESH}
            !
            ! The input field on this card indicates the size of the axial nodes in the
            ! core in cm. The input requires \fldlink[Core Geometry,TH_AXIAL]{nzpl} values.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{hz[1:nzpl]}{Integer}{Default = N/A}
            ! This card should be input as a 1D array of axial node sizes in cm. The array
            ! has a dimension of \fldlink[Core Geometry,TH_AXIAL]{nzpl}, where each element
            ! relates to a single fuel axial node in the core.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! The example below shows the AX_MESH card indicating 27 axial nodes with a size
            ! of 15.24 cm.
            !
            ! \begin{examplebox}
            ! !              hz
            ! AX_MESH       27*15.24
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(21) = card_info(21) + 1                      
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname,(fdum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (nzpl .NE. ndataf) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (MINVAL(fdum(1:nzpl)) .LT. 0.001 .OR. MAXVAL(fdum(1:nzpl)) .GT. 500.0) THEN
                        mesg='PATHS_hz values are outside of range [1e-3:500]'
                        CALL input_err(mesg,cardname)                        
                    END IF    
                    PATHS_hz(1:nzpl) = fdum(1:nzpl) / 100.0 ! convert to m
                CASE DEFAULT
                END SELECT
            CASE('TH_CONF') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="TH_CONF" description="Provides a radial map"
            ! "between PARCS assemblies and PATHS channels.">
            ! \Card{TH_CONF}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! This card sets the map of channel numbers between PARCS assemblies and
            ! \ac{PATHS} channels based on their radial location.
            !
            ! \note{For standalone \ac{PATHS} calculations, the radial location of each assembly will
            ! not influence the results, but must still be provided.}
            !
            ! \fldhdr{thconf[1:nasyx,1:nasyy]}{Real}{Default = N/A}
            !
            ! This card should be input as a 2D array with a total number of non-zero entries set
            ! by the input for \fldlink[Core Geometry,NK_RADIAL]{nfbxy} on card \cardref[Core Geometry]{NK_RADIAL}
            ! and the number of rows set by the \fldlink[Core Geometry,NK_RADIAL]{nrowy} input on
            ! \cardref[Core Geometry]{NK_RADIAL}.
            !            
            ! \note{Zeros may be given with Hexagonal layouts.}            
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! In the example below, \fldlink[Core Geometry,NK_RADIAL]{nfbxy} would be 9, and
            ! \fldlink[Core Geometry,NK_RADIAL]{nrowy} would be 3
            !
            ! \begin{examplebox}
            ! TH_CONF
            !    1   2   3
            !    4   5   6
            !    7   8   9
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(22) = card_info(22) + 1                                          
                CASE(1)
                    IF(coor .EQ. 1) THEN
                       DO j=1,nrowy
                          CALL read1more(indev,oneline,ndatafnew,cardname)
                          READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                          IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                          DO i=1,ndatafnew
                             IF(PATHS_nchan<idum(i))PATHS_nchan=idum(i)
                          ENDDO
                       ENDDO
                    ELSEIF(coor .EQ. 3)THEN
                       DO j=1,nrowy
                          CALL read1more(indev,oneline,ndatafnew,cardname)
                          READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                          IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                          k = 0
                          hcols(j,1) = ndatafnew
                          DO i=1,ndatafnew
                             IF(PATHS_nchan<idum(i))PATHS_nchan=idum(i)
                             IF(idum(i) .GT. 0) hcols(j,2) = hcols(j,2) + 1
                          ENDDO
                       ENDDO
                    ENDIF
                    IF (PATHS_nchan .GT. nchfb) THEN
                        mesg='found more radial channel than nchfb'
                        CALL input_err(mesg,cardname)                        
                    END IF
                CASE(2)
                    IF(coor .EQ. 1) THEN
                       ks = 0
                       ke = 0
                       DO ja=1,nrowy
                          CALL read1more(indev,oneline,ndatafnew,cardname)
                          ks = ke + 1
                          ke = ks + ndatafnew - 1
                          READ(oneline,*,IOSTAT=ios)(PATHS_conf(k),k=ks,ke)
                          IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                       ENDDO
                       IF (ke .NE. nfbxy) THEN
                          mesg='incorrect entries compared to nfbxy'
                          CALL input_err(mesg,cardname)                            
                       END IF
                       IF (MINVAL(PATHS_conf) .LT. 1 .OR. MAXVAL(PATHS_conf) .GT. nchfb) THEN
                           mesg='PATHS_conf out of range [1:nchfb]'
                           CALL input_err(mesg,cardname)                            
                       END IF                       
                    ELSE IF (coor .EQ. 3) THEN
                        k = 0
                        DO j=1,nrowy
                           CALL read1more(indev,oneline,ndatafnew,cardname)
                           READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                           IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                           DO i=1,ndatafnew
                              IF(idum(i) .GT. 0) THEN
                                  k = k + 1
                                  PATHS_conf(k) = idum(i)
                              END IF
                           ENDDO
                        ENDDO
                        IF (k .NE. nfbxy) THEN
                            mesg='incorrect entries compared to nfbxy'
                            CALL input_err(mesg,cardname)                            
                        END IF                        
                        IF (MINVAL(PATHS_conf) .LT. 1 .OR. MAXVAL(PATHS_conf) .GT. nchfb) THEN
                            mesg='PATHS_conf out of range [1:nchfb]'
                            CALL input_err(mesg,cardname)                            
                        END IF                         
                    ENDIF
                CASE DEFAULT
                END SELECT
            CASE('CH_TYPE')  ! required OR CH_T_MAP
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="CH_TYPE" description="Sets the channel type"
            ! "for each of the channels in the PATHS input model.">
            ! \Card{CH_TYPE}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! These cards may be used as an alternative to the \cardlink{CH_T_MAP} card.
            !
            ! Each entry of this card sets the type of a single channel, so a separate CH_TYPE card
            ! must be input for each channel. The number of card entries is set by the product of
            ! nasyx by nasyy. There must be one CH_TYPE card for each of those nasyx by nasyy entries.
            !
            ! \fldhdr{chtypeid}{Integer}{Default = N/A}
            ! This input field is the (numeric) channel ID number.
            !
            ! \fldhdr{chantype}{Integer}{Default = N/A}
            ! This field is the channel type corresponding to channel number \fldlink{chtypeid}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! In this example, there are two channels of type 1, two channels of type 2, and one
            ! channel of type 3. This example shows the input that will result in the same channel
            ! types as the example given for card \cardlink{CH_T_MAP}.
            !
            ! \begin{examplebox}
            ! !         chtypeid   chantype
            ! CH_TYPE      1        1
            ! CH_TYPE      2        1
            ! CH_TYPE      3        2
            ! CH_TYPE      4        2
            ! CH_TYPE      5        3
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(23) = card_info(23) + 1                     
                CASE(1)
                    READ(oneline,*,IOSTAT=ios) cardname,iPATHS_nchan,iPATHS_nchantype
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 2) THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)                                                  
                    END IF
                    PATHS_nchantype=MAX(PATHS_nchantype,iPATHS_nchantype)
                CASE(2)
!                    READ(oneline,*,IOSTAT=ios) cardname,chtypeid,PATHS_chantype(chtypeid)
                    READ(oneline,*,IOSTAT=ios) cardname,iPATHS_nchan,iPATHS_nchantype
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (PATHS_chantype(iPATHS_nchan) .NE. 0) THEN
                        mesg='channel type duplicate, please check entries'
                        CALL input_err(mesg,cardname)                                                  
                    ELSE
                        PATHS_chantype(iPATHS_nchan) = iPATHS_nchantype
                    END IF
                    IF (iPATHS_nchan .LT. 1 .OR. iPATHS_nchan .GT. nchfb) THEN
                        mesg='chantype out of range'
                        CALL input_err(mesg,cardname)                          
                    END IF         
                    IF (iPATHS_nchantype .LT. 1 .OR. iPATHS_nchantype .GT. nchfb) THEN
                        mesg='chantype out of range'
                        CALL input_err(mesg,cardname)                          
                    END IF                       
                CASE DEFAULT
                END SELECT
            CASE('CH_T_MAP') ! required OR CH_TYPE
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="CH_T_MAP" description="A radial map of channel"
            ! "types.">
            ! \Card{CH_T_MAP}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! This card gives a radial map of channel types. Therefore, the CH_T_MAP card
            ! must have the same dimensions as the \cardlink{TH_CONF} card.
            !
            ! \fldhdr{chantype[1:nasyx,1:nasyy]}{Integer}{Default = N/A}
            !
            ! This card should be input as a 2D array with a total number of non-zero entries set
            ! by the input for \fldlink[Core Geometry,NK_RADIAL]{nfbxy} on card \cardref[Core Geometry]{NK_RADIAL}
            ! and the number of rows set by the \fldlink[Core Geometry,NK_RADIAL]{nrowy} input on
            ! \cardref[Core Geometry]{NK_RADIAL}.
            !
            ! The channel types will be assigned to the channel IDs in the corresponding radial
            ! location specified on the \cardlink{TH_CONF} card.
            !
            ! \note{This card is an alternative to using \cardlink{CH_TYPE} cards. If CH_T_MAP
            ! is used, the \cardlink{CH_TYPE} cards should be omitted, and vice-versa.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! In the example below, \fldlink[Core Geometry,NK_RADIAL]{nfbxy} would be 5, and
            ! \fldlink[Core Geometry,NK_RADIAL]{nrowy} would be 1
            !
            ! \begin{examplebox}
            ! CH_T_MAP
            !    0   0   0   0   0
            !    0   1   1   1   0
            !    0   1   2   1   0
            !    0   1   1   1   0
            !    0   0   0   0   0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(24) = card_info(24) + 1                     
                CASE(1)
                    IF(coor .EQ. 1 .OR. coor .EQ. 3) THEN
                       DO j=1,nrowy
                          CALL read1more(indev,oneline,ndatafnew,cardname)
                          READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                          IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                          DO i=1,ndatafnew
                             IF(PATHS_nchantype<idum(i))PATHS_nchantype=idum(i)
                          ENDDO
                       ENDDO
                    ENDIF
                CASE(2)
                    PATHS_chantype=0
                    IF(coor .EQ. 1) THEN
                        ks = 0
                        ke = 0
                        DO ja=1,nrowy
                            CALL read1more(indev,oneline,ndatafnew,cardname)
                            ks = ke + 1
                            ke = ks + ndatafnew - 1
                            READ(oneline,*,IOSTAT=ios)(PATHS_ct_map(k),k=ks,ke)
                            IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                        ENDDO
                        IF (ke .NE. nfbxy) THEN
                            mesg='incorrect entries compared to nfbxy'
                            CALL input_err(mesg,cardname)                            
                        END IF
                        DO k=1,ke
                            chan=PATHS_conf(k)
                            IF(PATHS_chantype(chan)==0)THEN
                                PATHS_chantype(chan)=PATHS_ct_map(k)
                            ELSEIF(PATHS_chantype(chan).NE.PATHS_ct_map(k))THEN
                                mesg='channel type mismatch, grouped chan has different type or th_conf duplicate'
                                CALL input_err(mesg,cardname)  
                            ENDIF
                        ENDDO
                        IF (MINVAL(PATHS_chantype) .LT. 0 .OR. MAXVAL(PATHS_chantype) .GT. PATHS_nchantype) THEN
                            mesg='PATHS_chantype out of range [0:nchfb]'
                            CALL input_err(mesg,cardname)                            
                        END IF                          
                    ELSEIF (coor .EQ. 3) THEN
                        k = 0
                        DO j=1,nrowy
                            CALL read1more(indev,oneline,ndatafnew,cardname)
                            READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                            IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                            DO i=1,ndatafnew
                                IF(idum(i) .GT. 0) THEN
                                    k = k + 1
                                    PATHS_ct_map(k) = idum(i)
                                END IF
                            ENDDO
                        ENDDO
                        IF (k .NE. nfbxy) THEN
                            mesg='incorrect entries compared to nfbxy'
                            CALL input_err(mesg,cardname)                            
                        END IF
                        DO j=1,k
                            chan=PATHS_conf(j)
                            IF(PATHS_chantype(chan)==0)THEN
                                PATHS_chantype(chan)=PATHS_ct_map(j)
                            ELSEIF(PATHS_chantype(chan).NE.PATHS_ct_map(j))THEN
                                mesg='channel type mismatch, grouped chan has different type or th_conf duplicate'
                                CALL input_err(mesg,cardname)  
                            ENDIF
                        ENDDO
                        IF (MINVAL(PATHS_chantype) .LT. 0 .OR. MAXVAL(PATHS_chantype) .GT. PATHS_nchantype) THEN
                            mesg='PATHS_chantype out of range [0:nchfb]'
                            CALL input_err(mesg,cardname)                            
                        END IF                        
                    ENDIF
                CASE DEFAULT
                END SELECT
            CASE('CHAN_GEOM')  ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="CHAN_GEOM" description="Input detailed channel geometry"
            ! "data.">
            ! \Card{CHAN_GEOM}
            !
            ! This card provides detailed information about each channel in the \ac{PATHS} model. This includes
            ! the channel ID number, flow area, hydraulic diameter, etc.
            !
            ! \note{A separate CHAN_GEOM card must be provided for each channel type specified in the
            ! \cardlink{CH_TYPE} or \cardlink{CH_T_MAP} cards.}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{chgeomid}{Integer}{Default = N/A}
            ! The channel ID number.
            !
            ! \fldhdr{achan}{Real}{Default = N/A}
            ! Active flow area ($m^2$). \textit{Omit this input if acard on card \cardlink{AREA_CARD} is T.}
            !
            ! Only abrupt area changes are allowed, and these area changes will be treated at the
            ! edges between adjacent cell volumes, not within the volumes themselves. Since \ac{PATHS}
            ! calculates all flow quantities at cell edges, discontinuities in these values arising
            ! from abrupt area changes are handled internally in a mathematically consistent manner;
            ! however, only the values at the upstream ``side'' of the discontinuity are printed in
            ! the code output. The user should be aware of this convention, especially when
            ! extracting velocity values from the code.
            !
            ! \fldhdr{dhz}{Real}{Default = N/A}
            ! Hydraulic diameter (m). \textit{Omit this input if acard on card \cardlink{AREA_CARD} is T.}
            !
            ! \fldhdr{rgh}{Real}{Default = N/A}
            ! Wall roughness (m).
            !
            ! \fldhdr{pit}{Real}{Default = N/A}
            ! Pin pitch (m).
            !
            ! \fldhdr{nrod}{Integer}{Default = N/A}
            ! Number of fuel rods. \textit{Omit this input if acard on card \cardlink{AREA_CARD} is T.}
            !
            ! \fldhdr{rrod}{Real}{Default = N/A}
            ! Outer radius of fuel rod (including cladding - m).
            !
            ! \fldhdr{rf}{Real}{Default = N/A}
            ! Fuel pellet radius (m).
            !
            ! \fldhdr{tc}{Real}{Default = N/A}
            ! Clad thickness (m).
            !
            ! \fldhdr{hgap}{Real}{Default = N/A}
            ! Gap conductance ($W/m^2 -K$).
            !
            ! \note{Burnup- and temperature-dependent gap conductance is not available at this time.}
            !
            ! \fldhdr{nwrod}{Integer}{Default = N/A}
            ! Number of water rods.
            !
            ! \fldhdr{rwrod}{Real}{Default = N/A}
            ! Outer radius of water rod (m).
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! If \fldlink[Channel Geometry,AREA_CARD]{acard}=F:
            !
            ! \begin{examplebox}[long]
            ! !      chgeomid achan  dhz   rgh   pit  nrod rrod    rf     tc     hgap nwrod rwrod
            ! CHAN_GEOM  1    0.010 0.012 1.0e-6 0.01 81   5.0e-3 4.1e-3 6.0e-4 6000.0  1   1.8e-2
            ! CHAN_GEOM  2    0.011 0.013 1.0e-6 0.01 64   6.0e-3 4.4e-3 7.0e-4 6000.0  0   0.0
            ! CHAN_GEOM  3    0.012 0.015 1.0e-6 0.02 81   5.4e-3 3.9e-3 6.5e-4 6000.0  1   1.8e-2
            ! \end{examplebox}
            !
            ! An example for \fldlink[Channel Geometry,AREA_CARD]{acard}=T is shown below. When
            ! \fldlink[Channel Geometry,AREA_CARD]{acard}=T, the \fldlink[Channel Geometry,CHAN_GEOM]{achan},
            ! \fldlink[Channel Geometry,CHAN_GEOM]{dhz}, and \fldlink[Channel Geometry,CHAN_GEOM]{nrod} inputs
            ! must be omitted. Thus those fields are not included in the example.
            !
            ! \begin{examplebox}[long]
            ! !         chgeomid   rgh    pit   rrod    rf     tc     hgap  nwrod rwrod
            ! CHAN_GEOM     1     1.0e-6  0.01  5.0e-3 4.1e-3 6.0e-4 6000.0   1   1.8e-2
            ! CHAN_GEOM     2     1.0e-6  0.01  6.0e-3 4.4e-3 7.0e-4 6000.0   0   0.0
            ! CHAN_GEOM     3     1.0e-6  0.02  5.4e-3 3.9e-3 6.5e-4 6000.0   1   1.8e-2
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
               ! Aaron - removed achan, dhz, and nrod from CHAN_GEOM; added separate
               !        AREA_CHAN and NROD_CHAN card to input nodewise channel area
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(25) = card_info(25) + 1                     
                CASE(2)
                    IF (PATHS_acard) THEN ! if AREA_CHAN card will be provided, no need for area on CHAN_GEOM card
                       IF (ndataf .NE. 9) THEN
                           mesg='incorrect number of entries given'
                           CALL input_err(mesg,cardname)                          
                       END IF                          
                       READ(oneline,*,IOSTAT=ios) cardname,chgeomid,             &
                            PATHS_rgh(chgeomid),PATHS_pit(chgeomid),          &
                            PATHS_rrod(chgeomid),PATHS_rf(chgeomid),          &
                            PATHS_tc(chgeomid),PATHS_kgap(chgeomid),          &
                            PATHS_nwrod(chgeomid),PATHS_rwrod(chgeomid)
                       IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE
                       IF (ndataf .NE. 12) THEN
                           mesg='incorrect number of entries given'
                           CALL input_err(mesg,cardname)                          
                       END IF                        
                       READ(oneline,*,IOSTAT=ios) cardname,chgeomid,fdum(1),     &
                            fdum(2),PATHS_rgh(chgeomid),PATHS_pit(chgeomid),  &
                            fdum(3),PATHS_rrod(chgeomid),PATHS_rf(chgeomid),  &
                            PATHS_tc(chgeomid),PATHS_kgap(chgeomid),          &
                            PATHS_nwrod(chgeomid),PATHS_rwrod(chgeomid)
                       IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                       DO j=1,nzpl+nchm
                          PATHS_achan(j,chgeomid)=fdum(1)
                          PATHS_dhz(j,chgeomid)=fdum(2)
                          PATHS_nrod(j,chgeomid)=fdum(3)
                       ENDDO
                    ENDIF
! Some algebra
                    PATHS_kgap(chgeomid)=PATHS_kgap(chgeomid)*                          &
                         (PATHS_rrod(chgeomid)-PATHS_tc(chgeomid)-PATHS_rf(chgeomid))
                CASE DEFAULT
                END SELECT
            CASE('KA_CHAN') ! required
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="KA_CHAN" description="The vector of frictional"
            ! "loss factors.">
            ! \Card{KA_CHAN}
            !
            ! This card is the vector of frictional loss factors for each of the
            ! nz axial levels specified on the GRID_Z card of the GEOM block in the PARCS
            ! input \cite{PARCSUserV2}. One vector is provided for each channel type.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{chgeomid}{Integer}{Default = N/A}
            ! Channel type ID number.
            !
            ! \fldhdr{kloss[1:nz]}{Real}{Default = 0.0}
            ! 1D Vector of frictional loss factors. Input the same number of frictional loss
            ! factors as number of axial levels specified on the GRID_Z card of the PARCS input.
            !
            ! \note{Users are suggested to give the first value of at least 5.0. Some instability
            ! can be encountered when smaller values are used.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !       chgeomid  kloss
            ! KA_CHAN   1       203.12   26*0.0  1.26
            ! KA_CHAN   2       33.17    26*0.0  1.26
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(26) = card_info(26) + 1                     
                CASE(2)
                    IF (ndataf-1 .NE. nzpl+nchm) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                    READ(oneline,*,IOSTAT=ios) cardname,chgeomid,(PATHS_kloss(i,chgeomid),i=1,nzpl+nchm)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                CASE DEFAULT
                END SELECT
            CASE('AREA_CARD')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="AREA_CARD" description="Option to allow for"
            ! "axially varying channel geometry.">
            ! \Card{AREA_CARD}
            !
            ! This card is used to select between single inputs for the channel axial geometry
            ! or multiple inputs, allowing for axial variation.
            !
            ! The \cardlink{AREA_CHAN} and \cardlink{HD_CHAN} cards are required input if the value
            ! on this card is TRUE.
            !
            ! The number of required inputs on the \cardlink{CHAN_GEOM} card depends on
            ! the input on this card.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{acard}{Boolean}{Default = \default}
            ! Axially varying channel geometry option:
			!
            ! \begin{fldoptions}
            !    \fldopt{T} Values for \fldlink[Channel Geometry,AREA_CHAN]{achan}, \fldlink[Channel Geometry,HD_CHAN]{hd},
            !    and \fldlink[Channel Geometry,NROD_CHAN]{nrod} are entered as arrays on the \cardlink{AREA_CHAN},
            !    \cardlink{HD_CHAN}, and \cardlink{NROD_CHAN} cards instead of the \cardlink{CHAN_GEOM} card.
            !    \fldopt{F} Values for \fldlink[Channel Geometry,CHAN_GEOM]{achan}, \fldlink[Channel Geometry,CHAN_GEOM]{dhz},
            !    and \fldlink[Channel Geometry,CHAN_GEOM]{nrod} are expected to be input in the \cardlink{CHAN_GEOM} card.
            ! \end{fldoptions}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !               acard
            ! AREA_CARD        F
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
               ! Aaron - added AREA_CARD; defines whether chan areas are inputted
               !  on CHAN_GEOM card or on separate AREA_CHAN card
               ! (This is to allow backward compatibility w/ old inputs; but this
               !  might eventually change to require separate AREA_CHAN card in the future)
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(27) = card_info(27) + 1                     
                CASE(1,2)
                    READ(oneline,*,IOSTAT=ios) cardname,PATHS_acard
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (ndataf .NE. 1) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                CASE DEFAULT
                END SELECT
            CASE('AREA_CHAN')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="AREA_CHAN" description="The array of channel areas"
            ! "for each channel type.">
            ! \Card{AREA_CHAN}
            !
            ! This card provides the channel areas for each axial cell for each channel type.
            !
            ! Only abrupt area changes are allowed, and these area changes will be treated at the
            ! edges between adjacent cell volumes, not within the volumes themselves. Since \ac{PATHS}
            ! calculates all flow quantities at cell edges, discontinuities in these values arising
            ! from abrupt area changes are handled internally in a mathematically consistent manner;
            ! however, only the values at the upstream ``side'' of the discontinuity are printed in
            ! the code output. The user should be aware of this convention, especially when
            ! extracting velocity values from the code.
            !
            ! \note{This card is only input if variable acard=T on \cardref[Channel Geometry]{AREA_CARD}.}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{chgeomid}{Integer}{Default = N/A}
            ! Channel type ID number.
            !
            ! \fldhdr{achan[1:nz]}{Real}{Default = N/A}
            ! This input is the vector of channel areas for each of the
            ! nz axial cells specified on the GRID_Z card of the GEOM block in the PARCS
            ! input \cite{PARCSUserV2}. One vector is provided for each channel type (\fldlink[AREA_CHAN]{chgeomid}).
            !
            ! Array values are entered starting with the bottom axial node and ending with the top.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !          chgeomid  achan
            ! AREA_CHAN    1       28*0.01
            ! AREA_CHAN    2       10*0.01  12*0.015  6*0.020
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(28) = card_info(28) + 1                       
                CASE(2)
                    IF (ndataf-1 .NE. nzpl+nchm) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                    IF (PATHS_acard) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,chgeomid,(PATHS_achan(i,chgeomid),i=1,nzpl+nchm) 
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE
                        mesg='AREA_CHAN should not be input when AREA_CARD = F'
                        CALL input_err(mesg,cardname)                                                  
                    END IF
                    IF (MINVAL(PATHS_achan(1:nzpl+nchm,chgeomid)) .LT. 1e-10 .OR. &
                        MAXVAL(PATHS_achan(1:nzpl+nchm,chgeomid)) .GT. 10.0) THEN
                        mesg='PATHS_achan outside of range [1e-10:10]'
                        CALL input_err(mesg,cardname)                          
                    END IF                                         
                CASE DEFAULT
                END SELECT
            CASE('NROD_CHAN')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="NROD_CHAN" description="The number of fuel rods in"
            ! "each axial cell of a channel.">
            ! \Card{NROD_CHAN}
            !
            ! This card is a list array of the number of fuel rods present in each axial cell for
            ! a specified channel type. The array values are entered starting with the bottom axial node
            ! and ending with the top.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{chgeomid}{Integer}{Default = N/A}
            ! Channel type ID number.
            !
            ! \fldhdr{nrod[1:nz]}{Integer}{Default = N/A}
            ! This input is the vector specifying the number of fuel rods for each of the
            ! nz axial cells specified on the GRID_Z card of the GEOM block in the PARCS
            ! input \cite{PARCSUserV2}. One vector is provided for each channel type (\fldlink[NROD_CHAN]{chgeomid}).
            !
            ! \note{Even unheated nodes must have a non-zero value for nrod.}            
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !          chgeomid  nrod
            ! NROD_CHAN    1       28*100
            ! NROD_CHAN    2       10*100   12*90  6*80
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(29) = card_info(29) + 1                       
                CASE(2)
                    IF (ndataf-1 .NE. nzpl+nchm) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                    IF (PATHS_acard) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,chgeomid,(PATHS_nrod(i,chgeomid),i=1,nzpl+nchm)
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE
                        mesg='NROD_CHAN should not be input when AREA_CARD = F'
                        CALL input_err(mesg,cardname)                                                  
                    END IF
                    IF (MINVAL(PATHS_nrod(1:nzpl+nchm,chgeomid)) .LT.    0 .OR. &
                        MAXVAL(PATHS_nrod(1:nzpl+nchm,chgeomid)) .GT. 1000) THEN
                        mesg='PATHS_nrod outside of range [0:1000]'
                        CALL input_err(mesg,cardname)                          
                    END IF       
                    IF (MINVAL(PATHS_nrod(:,chgeomid)) .LT. 1) THEN
                        mesg='NROD_CHAN values must be >= 1 for all nodes'
                        CALL input_err(mesg,cardname)                         
                    END IF
                CASE DEFAULT
                END SELECT
            CASE('HD_CHAN')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="HD_CHAN" description="The array of channel hydraulic"
            ! "diameters.">
            ! \Card{HD_CHAN}
            !
            ! This card provides the channel areas for each axial cell for each channel type.
            !
            ! \note{This card is only input if variable acard=T on \cardref[Channel Geometry]{AREA_CARD}.}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{chgeomid}{Integer}{Default = N/A}
            ! Channel type ID number.
            !
            ! \fldhdr{hd[1:nz]}{Real}{Default = N/A}
            ! This input is the vector of hydraulic diameters for each of the
            ! nz axial cells specified on the GRID_Z card of the GEOM block in the PARCS
            ! input \cite{PARCSUserV2}. One vector is provided for each channel type (\fldlink[HD_CHAN]{chgeomid}).
            !
            ! Array values are entered starting with the bottom axial node and ending with the top.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !        chgeomid   hd
            ! HD_CHAN    1      28*0.08
            ! HD_CHAN    2      10*0.08  12*0.012  6*0.016
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(30) = card_info(30) + 1                       
                CASE(2)
                    IF (ndataf-1 .NE. nzpl+nchm) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                    IF (PATHS_acard) THEN
                        READ(oneline,*,IOSTAT=ios) cardname,chgeomid,(PATHS_dhz(i,chgeomid),i=1,nzpl+nchm)
                        IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ELSE
                        mesg='HD_CHAN should not be input when AREA_CARD = F'
                        CALL input_err(mesg,cardname)                                                  
                    END IF  
                    IF (MINVAL(PATHS_dhz(1:nzpl+nchm,chgeomid)) .LT. 1e-10 .OR. &
                        MAXVAL(PATHS_dhz(1:nzpl+nchm,chgeomid)) .GT. 1000) THEN
                        mesg='PATHS_dhz outside of range [1e-10:1000]'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                CASE DEFAULT
                END SELECT
            CASE('AREA_LOSS')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="AREA_LOSS" description="The array of abrupt area"
            ! "change loss factors.">
            ! \Card{AREA_LOSS}
            !
            ! This card provides the user-specified loss factors to account for pressure loss due to
            ! abrupt area changes for each channel type.
            !
            ! Unlike other axially-dependent values, the aloss values are specified at the edges
            ! between cell volumes, not within the volumes themselves. This means that (nz+1)
            ! values are required for each channel type, instead of nz.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{chgeomid}{Integer}{Default = N/A}
            ! Channel type ID number.
            !
            ! \fldhdr{aloss[1:nz+1]}{Real}{Default = 0.0}
            ! This input is the vector of optional user-specified loss factors that account for pressure
            ! drops due to abrupt area changes. Input a value for each of the nz+1 axial cells. The number
            ! of cells (nz) is specified on the GRID_Z card of the GEOM block in the PARCS input
            ! \cite{PARCSUserV2}. One vector is provided for each channel type (chgeomid).
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !         chgeomid  aloss
            ! AREA_LOSS  1        29*0.0
            ! AREA_LOSS  2        10*100  1*1.0  11*0.0  1*0.8  6*0.0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(31) = card_info(31) + 1                                           
                CASE(2)
                    IF (ndataf-1 .NE. nzpl+nchm+1) THEN
                        mesg='incorrect entries given'
                        CALL input_err(mesg,cardname)                          
                    END IF                     
                    READ(oneline,*,IOSTAT=ios) cardname,chgeomid,(PATHS_aloss(i,chgeomid),i=1,nzpl+nchm+1)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF (MINVAL(PATHS_aloss(1:nzpl+nchm+1,chgeomid)) .LT.    0 .OR. &
                        MAXVAL(PATHS_aloss(1:nzpl+nchm+1,chgeomid)) .GT. 1e4) THEN
                        mesg='PATHS_aloss outside of range [0:1e4]'
                        CALL input_err(mesg,cardname)                          
                    END IF                      
                CASE DEFAULT
                END SELECT

            CASE('ORIF_MAP')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="ORIF_MAP" description="Optional orifice loss factor input.">
            ! \Card{ORIF_MAP}
            !
            ! This card activates the optional orifice loss factor input. The input defines which orifice
            ! type corresponds to each \ac{T/H} channel, so that the correct loss from the \cardlink{ORIF_LOSS}
            ! card is applied to each channel.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{orifopt}{Integer}{Default = N/A}
            ! Optional orifice loss factor input:
			!
            ! \begin{fldoptions}
            !    \fldopt{0} No separate orifice loss factor values will be used. Any inlet orifice loss factors
            !    should be input directly on the \cardlink{KA_CHAN} card.
            !    \fldopt{1} Inlet orifice loss factors may be input on the \cardlink{ORIF_LOSS} card. These will
            !    be added to the loss values for the inlet node specified in the \cardlink{KA_CHAN} card.
            ! \end{fldoptions}
            !
            ! \fldhdr{oriftype[1:nasyx,1:nasyy]}{Integer}{Default = N/A}
            !
            ! This field should be input as a 2D array with a total number of non-zero entries set
            ! by the input for \fldlink[Core Geometry,NK_RADIAL]{nfbxy} on card \cardref[Core Geometry]{NK_RADIAL}
            ! and the number of rows set by the \fldlink[Core Geometry,NK_RADIAL]{nrowy} input on
            ! \cardref[Core Geometry]{NK_RADIAL}.
            !
            ! This 2D array specifies the orifice type corresponding to each \ac{T/H} channel. The type determines which
            ! value on the \cardlink{ORIF_LOSS} card will be applied to each channel.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! ORIF_MAP         1
            !       2  1  1  1  2
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
               ! Aaron - added ORIF_MAP to specify input type for orifice kloss factors
               ! First value = PATHS_orifopt
               ! PATHS_orifopt=0 means no separate orif input (taken from KA_CHAN card like normal)
               ! PATHS_orifopt=1 means input 2D map of orifice types, and will look
               !        for ORIF_LOSS cards for kloss for each orifice type in the 2d map
               ! PATHS_orifopt=2 means input 2D map of orifice loss factors directly (no separate ORIF_LOSS cards)
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(32) = card_info(32) + 1                                           
                CASE(1)
                    IF(coor .EQ. 1 .OR. coor .EQ. 3) THEN
                       DO j=1,nrowy
                          CALL read1more(indev,oneline,ndatafnew,cardname)
                          READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                          IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                          DO i=1,ndatafnew
                             IF(PATHS_numorif<idum(i))PATHS_numorif=idum(i)
                          ENDDO
                       ENDDO
                    ENDIF
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname,PATHS_orifopt
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF(PATHS_orifopt .LT.0 .OR. PATHS_orifopt .GT. 2) THEN
                        mesg='input must be [0:2]'
                        CALL input_err(mesg,cardname)   
                    END IF
                    IF(PATHS_orifopt.EQ.1) THEN
                       thconf=0
                       PATHS_oriftype=0
                       ks=0
                       ke=0
                       IF(coor .EQ. 1) THEN
                           DO ja=1,nrowy
                              CALL read1more(indev,oneline,ndatafnew,cardname)
                              ks = ke + 1
                              ke = ks + ndatafnew - 1
                              READ(oneline,*,IOSTAT=ios)(thconf(k),k=ks,ke)
                              IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                           ENDDO
                           IF (ke .NE. nfbxy) THEN
                               mesg='incorrect entries compared to nfbxy'
                               CALL input_err(mesg,cardname)                                    
                           END IF
                       ELSE IF (coor .EQ. 3) THEN
                           k = 0
                           DO j=1,nrowy
                              CALL read1more(indev,oneline,ndatafnew,cardname)
                              READ(oneline,*,IOSTAT=ios) (idum(i),i=1,ndatafnew)
                              IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                              DO i=1,ndatafnew
                                 IF(idum(i) .GT. 0) THEN
                                     k = k + 1
                                     thconf(k) = idum(i)
                                 END IF
                              ENDDO
                           ENDDO
                           IF (k .NE. nfbxy) THEN
                               mesg='incorrect entries compared to nfbxy'
                               CALL input_err(mesg,cardname)                                    
                           END IF                           
                       END IF
                       DO k=1,nfbxy
                           IF (thconf(k).GT.0) THEN
                              chan=PATHS_conf(k)
                              IF(PATHS_oriftype(chan)==0)THEN
                                 PATHS_oriftype(chan)=thconf(k)
!                                 IF(PATHS_oriftype(chan) > PATHS_numorif) PATHS_numorif = PATHS_oriftype(chan)
                              ELSEIF(PATHS_oriftype(chan).NE.thconf(k))THEN
                                  mesg='channel orifice mismatch, grouped chan has different type or th_conf duplicate'
                                  CALL input_err(mesg,cardname) 
                              ENDIF
                           ENDIF
                       END DO
                       IF (MINVAL(PATHS_oriftype) .LT. 0 .OR. MAXVAL(PATHS_oriftype) .GT. nchfb) THEN
                           mesg='PATHS_oriftype out of range [0:nchfb]'
                           CALL input_err(mesg,cardname)                            
                       END IF                         
                    ELSEIF(PATHS_orifopt.EQ.2) THEN
                       ! Not currently implemented
                    ENDIF
                CASE DEFAULT
                END SELECT
            CASE('ORIF_LOSS')
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Channel Geometry" card="ORIF_LOSS" description="The array of orifice loss"
            ! "factors for each orifice type.">
            ! \Card{ORIF_LOSS}
            !
            ! This card provides the orifice loss factors that will be added to the channels with the
            ! corresponding orifice type specified in the \cardlink{ORIF_MAP} card.
            !
            ! Entry 1 on the ORIF_LOSS card will be applied to orifice type 1, entry 2 to orifice type 2, etc.
            !
            ! The code will expect numorif entries, where numorif is the maximum
            ! orifice type value specified on the \cardlink{ORIF_MAP} card.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{orifloss[1:numorif]}{Real}{Default = 0.0}
            ! 1D array of optional orifice loss factors.
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !          orifloss
            ! ORIF_LOSS  30.7   242.2
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(33) = card_info(33) + 1                       
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname,(fdum(i),i=1,ndataf)
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF(ndataf.NE.PATHS_numorif) THEN
                        mesg='number of orifice values are out of range'
                        CALL input_err(mesg,cardname)  
                    ELSE
                        PATHS_orifloss = 0.0
                        DO i=1,PATHS_numorif
                           PATHS_orifloss(i) = fdum(i)
                        ENDDO
                        IF (MINVAL(PATHS_orifloss) .LT. 0 .OR. MAXVAL(PATHS_orifloss) .GT. 1e4) THEN
                            mesg='PATHS_orifloss out of range [0:1e4]'
                            CALL input_err(mesg,cardname)                            
                        END IF                          
                    ENDIF
                CASE DEFAULT
                END SELECT
            CASE('TH_USEDEP') ! required if standalone OR TH_RELP
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Power" card="TH_USEDEP" description="Selects the axial and"
            ! "radial power distribution for depletion calculations.">
            ! \Card{TH_USEDEP}
            !
            ! This card is used to select between an axial and radial distribution provided
            ! by a depletion file or on the \cardlink{TH_RELP} card.
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! \fldhdr{idep}{Boolean}{Default = N/A}
            ! Power shape option:
			!
            ! \begin{fldoptions}
            !    \fldopt{T} Use axial and radial power distribution from the depletion file
            !     in the \fldlink{FN_TEMP} field.
            !    \fldopt{F} Use axial and radial power distrution from the \cardlink{TH_RELP} card.
            ! \end{fldoptions}
            !
            ! \fldhdr{FN_TEMP}{String}{Default = N/A}
            ! Name of the depletion file.
            !
            ! \note{if \fldlink{idep} = F (false), FN_TEMP may be omitted.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! \begin{examplebox}
            ! !               idep      FN_TEMP
            ! TH_USEDEP        T       'parcs.dep'
            ! !               idep
            ! TH_USEDEP        F
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(34) = card_info(34) + 1                       
                CASE(2)
                    READ(oneline,*,IOSTAT=ios) cardname,ldum(1),paths_depfname
                    IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    IF(ndataf .NE. 2)THEN
                        mesg='incorrect number of entries'
                        CALL input_err(mesg,cardname)
                    END IF
                    IF (ldum(1)) pwr_inp = 2
                    INQUIRE(file = paths_depfname,EXIST=io_exist,IOSTAT=ios)
                    IF (.NOT. io_exist) THEN
                        mesg='DEP file cannnot be found'
                        CALL input_err(mesg,cardname)                        
                    ENDIF                    
                    INQUIRE(unit = fid_rlp, OPENED=io_open, IOSTAT=io_flg)
                    IF (io_open) CLOSE(fid_rlp)
                    CALL get_blnk_unit(fid_rlp)                    
                    OPEN(fid_rlp,file=paths_depfname,form='FORMATTED')
                CASE DEFAULT
                END SELECT
            CASE('TH_RELP') ! required if standalone OR TH_USEDEP
            ! ================================================================================
            ! WARNING: COMMENTS BETWEEN THE DOUBLE LINES ARE USED TO AUTOGENERATE THE
            !          DOCUMENTATION. DO NOT DELETE OR MODIFY UNLESS YOU KNOW WHAT YOU ARE
            !          DOING.
            !
            ! <latex cardgroup="Core Power" card="TH_RELP" description="Defines relative thermal"
            ! "power to each of the core channels.">
            ! \Card{TH_RELP}
            !
            ! %---------------------------------------
            ! \sectionhead{Input Fields}
            !
            ! This card defines the thermal power to all the levels in each of the channels.
            !
            ! \ac{PATHS} interprets relative power values in the sense of power per unit length.
            ! In other words, the total power provided to axial node j of channel i is
            ! proportional not to Prel(j,i) but to (Prel(j,i)*hz(j)).  This must
            ! be taken into account when providing relative power values for non-uniformly
            ! spaced axial meshes.
            !
            ! \ac{PATHS} scales the relative powers globally to give the correct total core power.
            ! However, the power peaking factor for each channel will depend on the sum of
            ! (Prel(j,i)*hz(j)) across all axial nodes for that channel.
            !
            ! \note{If \fldlink[Initialization,TH_IPATHS]{paths_onlyflag} = F or \fldlink{idep} = T,
            ! omit the TH_RELP card.  The relative power will instead be determined by PARCS
            ! neutronics or from the depletion file, respectively.}
            !
            ! \fldhdr{Prel[1:nz,1:nchan]}{Real}{Default = 0.0}
            !
            ! This field defines the thermal power to each of the axial levels (rows) for all channels (columns).
            ! The Prel array is a 2D array that is of dimension nz by nchan. The values for the axial
            ! power are provided from the top of the core with row 1 of the array to the core bottom with row nz.
            !
            ! The number of rows in the \fldlink[TH_RELP]{Prel} array is nz, which is equal to the
			! \fldlink[Core Geometry,TH_AXIAL]{nzpl} input plus 1 if \fldlink[Channel Geometry,CHIMNEY]{usechimney}
			! is true, and plus 0 otherwise.
			!
            ! The number of columns is nchan, which is equal to the largest value on the
            ! \cardref[Channel Geometry]{TH_CONF}
            !
            ! \note{The array input must be entered in nz lines with nchan values on each line.}
            !
            ! %---------------------------------------
            ! \sectionhead{Example}
            !
            ! In the example below, if we assume a uniform 0.1 m spacing for all axial
            ! nodes (i.e. hz(j)=0.1 for all j), then by comparing the sum of (Prel(j,i)*hz(j))
            ! across all axial nodes we obtain power peaking factors of 0.880, 0.579, 1.494,
            ! 1.055, and 0.991 for the five channels, respectively.  The entries in the table
            ! below could be scaled by any arbitrary factor (other than zero) and the solution
            ! would not change because the code internally normalizes the values globally to
            ! yield the correct total power.
            !
            ! \begin{examplebox}[long]
            ! TH_RELP
            ! !Prel
            ! 0             0             0             0             0
            ! 0             0             0             0             0
            ! 0.020725574   0.013169913   0.038682183   0.022062341   0.020951672
            ! 0.029385196   0.019303560   0.052128294   0.032873320   0.028526278
            ! 0.036214773   0.024688498   0.060789215   0.045392490   0.035435956
            ! 0.038589154   0.026646658   0.061378827   0.051848432   0.037553983
            ! 0.039423073   0.027407281   0.058373855   0.055002627   0.037680210
            ! 0.040064932   0.027602899   0.056688111   0.055716549   0.037179278
            ! 0.040527667   0.027313941   0.054697317   0.054214930   0.036898995
            ! 0.041429255   0.027480762   0.055663689   0.053573648   0.037752766
            ! 0.042237300   0.027747874   0.058899725   0.052879022   0.039070694
            ! 0.043257309   0.028202660   0.064483964   0.054068513   0.040756369
            ! 0.043386676   0.028195709   0.069757462   0.054237630   0.041521681
            ! 0.043661332   0.028343663   0.074193210   0.054798325   0.042281030
            ! 0.043774776   0.028328768   0.077738849   0.054478252   0.042521557
            ! 0.043304080   0.028122228   0.077996093   0.052842701   0.042018637
            ! 0.042602513   0.027765748   0.077373472   0.050556790   0.041082372
            ! 0.042344774   0.027675386   0.078032517   0.049514850   0.041361661
            ! 0.041058071   0.026877029   0.075766268   0.046272123   0.040481055
            ! 0.040026121   0.026248472   0.074551758   0.044064527   0.041143995
            ! 0.038324449   0.025160164   0.072537061   0.041255375   0.047334081
            ! 0.035858517   0.023516780   0.068265220   0.037480329   0.059060258
            ! 0.031493876   0.020583513   0.060716367   0.031850675   0.061904835
            ! 0.027225763   0.017469523   0.053096942   0.026722696   0.057212575
            ! 0.020288712   0.012629731   0.040424839   0.019331505   0.046081751
            ! 0.014597563   0.008921442   0.032068969   0.013966309   0.035676483
            ! 0             0             0             0             0
            ! 0             0             0             0             0
            ! \end{examplebox}
            ! </latex>
            ! ================================================================================
                SELECT CASE(read_opt)
                CASE(0)
                    card_info(35) = card_info(35) + 1                     
                CASE(2)
                    DO j=1,nzpl
                       CALL read1more(indev,oneline,ndatafnew,cardname)
                       IF (ndatafnew .NE. PATHS_nchan) THEN
                           mesg='incorrect number of entries'
                           CALL input_err(mesg,cardname)                           
                       END IF
                       pwr_inp = 1
                       READ(oneline,*,IOSTAT=ios)(PATHS_Prel(j,i),i=1,PATHS_nchan)
                       IF (ios .NE. 0) CALL iostat_err(inp_opt,cardname)
                    ENDDO                    
                CASE DEFAULT
                END SELECT                
            CASE default
            END SELECT
            ! ----------------------------------------------------------------------
            DO i=1,ndataf
               idum(i)=0
               fdum(i)=0.0
               ldum(i)=.FALSE.
            ENDDO
!         ENDDO
!         !
!         EXIT rloop
         !
      END DO rloop
      !
! Tests after check phase, only cards
      IF (read_opt .EQ. 0) THEN
          inp_test = 0
! Do checks only if PATHS input (ignore when coupled)
          IF (card_info(2) .EQ. 1) THEN
              ! First check for CH_TYPE of CH_T_MAP
              ! If passes test, continue checks
              IF (card_info(23) .LT. 1 .AND. card_info(24) .LT. 1) THEN
                  WRITE(fid_out,*)'Either ',inp_card(23),' or ',inp_card(24),' is required'
                  mesg='Required input was not given, check input'
                  CALL input_stp(mesg)                                                       
              ELSEIF (card_info(23) .GT. 0 .AND. card_info(24) .GT. 0) THEN   
                  WRITE(fid_out,*)'Inputs ',inp_card(23),' and ',inp_card(24),' are not compatiable'
                  mesg='Required input was not given, check input'
                  CALL input_stp(mesg)                                                       
              END IF
              ! Check for required inputs
              DO inp_test=17,22,1
                  IF (card_info(inp_test) .NE. 1) THEN
                      WRITE(fid_out,*)'Input ',inp_card(inp_test),' was not found'
                      mesg='Required input was not given, check input'
                      CALL input_stp(mesg)                                         
                  END IF
              END DO
              DO inp_test=25,26,1
                  IF (card_info(inp_test) .LT. 1) THEN
                      WRITE(fid_out,*)'Input ',inp_card(inp_test),' was not found'
                      mesg='Required input was not given, check input'
                      CALL input_stp(mesg)                                         
                  END IF
              END DO              
          ELSE IF (card_info(2) .GT. 1) THEN 
              mesg='multiple instances of same PATHS input card'
              CALL input_stp(mesg)
          END IF    
      END IF
      IF (read_opt .EQ. 0 .AND. card_info(2) .EQ. 1) THEN          
          IF (paths_onlyflag) THEN
              WRITE(fid_out,*)'Running PATHS in stand alone mode'
          ELSE IF (.NOT. paths_onlyflag) THEN
              WRITE(fid_out,*)'Running PATHS coupled to PARCS'
              IF (card_info(4) .LT. 1) THEN
                  mesg='COMM_PARM is required when coupled'
                  CALL input_stp(mesg)            
              END IF
          END IF
          IF (card_info(34) .GT. 0 .AND. card_info(35) .GT. 0) THEN   
              WRITE(fid_out,*)'Inputs ',inp_card(34),' and ',inp_card(35),' are not compatiable'
              mesg='Error found in input'
              CALL input_stp(mesg)                                                                 
          END IF    
          IF (paths_onlyflag) THEN
              IF (card_info(34) .LT. 1 .AND. card_info(35) .LT. 1) THEN
                  WRITE(fid_out,*)'Either ',inp_card(34),' or ',inp_card(35),' is required'
                  mesg='Required input was not given, check input'
                  CALL input_stp(mesg)                                                       
              END IF                        
          ELSE
              IF (card_info(34) .GT. 0) THEN
                  WRITE(fid_out,*)'Card',inp_card(34),' will be used for power initial guess'
              END IF               
              IF (card_info(35) .GT. 0) THEN
                  WRITE(fid_out,*)'Card',inp_card(35),' will be used for power initial guess'
              END IF                             
          END IF           
          ! Normalized inputs to check duplicates
          IF (card_info(23) .GT. 0) card_info(23) = card_info(23) / card_info(23)
          IF (card_info(25) .GT. 0) card_info(25) = card_info(25) / card_info(25)
          IF (card_info(26) .GT. 0) card_info(26) = card_info(26) / card_info(26)
          !
          IF (card_info(28) .GT. 0) card_info(28) = card_info(28) / card_info(28)          
          IF (card_info(29) .GT. 0) card_info(29) = card_info(29) / card_info(29)          
          IF (card_info(30) .GT. 0) card_info(30) = card_info(30) / card_info(30)          
          IF (card_info(31) .GT. 0) card_info(31) = card_info(31) / card_info(31)   
          DO inp_test=1,35
              IF (card_info(inp_test) .GT. 1) THEN
                   WRITE(fid_out,*)'Input ',inp_card(inp_test),' was found more than 1x'
                   mesg='Duplicate input was given, check input'
                   CALL input_stp(mesg)
              END IF
          END DO    
          IF (SUM(card_info) .GT. 35) THEN
              mesg='too many cards have been given'
              CALL input_stp(mesg)                          
          END IF          
      END IF
      IF (read_opt .EQ. 0) THEN
          CLOSE(fid_out)
      END IF          
      !
      RETURN
      

CONTAINS

     SUBROUTINE iostat_err(iopt,cname)

         USE PATHS_varcpld, ONLY: fid_out, fid_scn

         IMPLICIT NONE

         CHARACTER(len=4),  INTENT(IN) :: iopt
         CHARACTER(len=10), INTENT(IN) :: cname

         WRITE(fid_scn,*)'Error PATHS input: ',cname,' during the ', iopt,' attempt.'
         WRITE(fid_scn,*)'Check input for card ', cname         
         
         IF (fid_out .NE. 0) THEN
             WRITE(fid_out,*)'Error PATHS input: ',cname,' during the ', iopt,' attempt.'
             WRITE(fid_out,*)'Check input for card ', cname
         END IF    
         STOP

     END SUBROUTINE iostat_err

     SUBROUTINE input_err(msg,cname)

         USE PATHS_varcpld, ONLY: fid_out, fid_scn

         IMPLICIT NONE

         CHARACTER(len=40),  INTENT(IN) :: msg
         CHARACTER(len=10), INTENT(IN) :: cname

         WRITE(fid_scn,*)'Err (PATHS): check input for card ',cname
         WRITE(fid_scn,*) msg         
         
         IF (fid_out .NE. 0) THEN
             WRITE(fid_out,*)'Err (PATHS): check input for card ',cname
             WRITE(fid_out,*) msg
         END IF    
         STOP

     END SUBROUTINE input_err
     
     SUBROUTINE input_stp(msg)

         USE PATHS_varcpld, ONLY: fid_out, fid_scn

         IMPLICIT NONE

         CHARACTER(len=40),  INTENT(IN) :: msg

         WRITE(fid_scn,*) 'Err (PATHS): ',msg         
         
         IF (fid_out .NE. 0) WRITE(fid_out,*) 'Err (PATHS): ',msg
         STOP

     END SUBROUTINE input_stp

END SUBROUTINE PATHS_read

