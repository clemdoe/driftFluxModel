!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_fileIOM

    USE IntrType, ONLY: sdk, sik
    USE PATHS_thvarM, ONLY: inpfile, outfile
    USE PATHS_varcpld, ONLY: fid_inp, fid_out, paths_id, get_blnk_unit
    USE PATHS_util, ONLY: mesg, term_execution

!    USE f90_unix, ONLY: getarg  ! This line is for Nag compiler only            
    
    IMPLICIT NONE

    LOGICAL :: ifinp
    
    CONTAINS

    SUBROUTINE PATHS_get_cmd_arg

        IMPLICIT NONE

        CALL getarg(1,inpfile)

    END SUBROUTINE PATHS_get_cmd_arg

    SUBROUTINE PATHS_inpfile_open

        IMPLICIT NONE

        INQUIRE(file=inpfile,exist=ifinp)
        
        IF (.NOT. ifinp) THEN
            mesg = 'ERROR: Input file not found'
            CALL term_execution(mesg)
        END IF    

        ! Open, uses getarg, but overwritten by interface
        CALL get_blnk_unit(fid_inp)
        OPEN(fid_inp,file=inpfile,status='old',position='rewind')

    END SUBROUTINE PATHS_inpfile_open


    SUBROUTINE PATHS_outfile_open
          
        IMPLICIT NONE

        ! Call and open outfile 
        CALL PATHS_set_output
        CALL get_blnk_unit(fid_out)
        OPEN(fid_out,file=outfile,status='replace')
        
        WRITE(*,*) 'PATHS Version - 1.05 [Aug 2017]'
        WRITE(fid_out,*) 'PATHS Version - 1.05 [Aug 2017]'
    
    END SUBROUTINE PATHS_outfile_open

    SUBROUTINE PATHS_inpfile_close
          
        IMPLICIT NONE

        ! Close the input file
        CLOSE(fid_inp)
    
    END SUBROUTINE PATHS_inpfile_close

    SUBROUTINE PATHS_outfile_close
          
        IMPLICIT NONE
        
        WRITE(*,*) 'PATHS Calculation Completed'
        WRITE(fid_out,*) 'PATHS Calculation Completed'        

        ! Close output file
        CLOSE(fid_out)
    
    END SUBROUTINE PATHS_outfile_close

    SUBROUTINE PATHS_set_output
    
        IMPLICIT NONE
       
! generic indexing variables
        INTEGER(sik) :: iend,i5,i,j

! Open actual output file
        DO i=1,80
            j=81-i
            if(inpfile(j:j)/=' ')then
!            IF(paths_id(j:j)/=' ')THEN
               iend=j
               EXIT
            END IF
        END DO
        DO i=0,iend-2
            j=iend-i
            if(inpfile(j:j) .EQ. '.')then
!            IF (paths_id(j:j) .EQ. '.') THEN
               iend=j-1
               EXIT
            END IF
        END DO
        i5=iend+10
        outfile = ' '
        outfile(1:iend)=inpfile(1:iend)  
!        outfile(1:iend)=paths_id(1:iend)  
        outfile = TRIM(outfile)
        outfile(iend+1:i5)='.paths_out'
        outfile = TRIM(outfile)

    END SUBROUTINE PATHS_set_output

END MODULE PATHS_fileIOM

