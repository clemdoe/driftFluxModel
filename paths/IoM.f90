MODULE Io
!
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NRC Prime Contract no. NRC-04-97-046 (Task #2)
!
!
!     BEGIN MODULE USE
      USE IntrType, ONLY: sdk, sik
!
      IMPLICIT NONE
      CHARACTER(LEN=40) :: stEosFName = 'trch2o'
      INTEGER(sik), SAVE :: iout = 36
      INTEGER(sik), PARAMETER :: stEosFIn = 66
      INTEGER(sik), DIMENSION(3) :: outUnt = (/ 7, 36, 6 /)
END MODULE Io
