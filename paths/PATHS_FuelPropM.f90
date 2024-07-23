!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_FuelProbM
      ! 
      !
      ! Linsen LI 06/07/2010
      USE IntrType, ONLY: sdk, sik
      USE PATHS_thvarM, ONLY: num_kfuel, dat_kfuel, &
                              num_kclad, dat_kclad

      IMPLICIT NONE

CONTAINS

      ! functions for thermal properties of fuel regions
      FUNCTION fkf(t)
      
            IMPLICIT NONE

            ! 4-th polynomial for thermal conductivity of uo2 in w/m-C, t in K

            INTEGER(sik) :: n
            REAL(sdk) :: fkf, t

            IF (num_kfuel .GT. 0) THEN
                fkf = 0.0
                DO n=1,num_kfuel
                    fkf = fkf + dat_kfuel(n) * t ** (n-1)
                END DO
            ELSE
                fkf=0.0000000000023595*t**4 -0.000000015236*t**3 +0.000036285*t**2 - 0.038352*t +17.94
            END IF
            !fkf=0.000000000002*t**4 - 2E-08*t**3 + 4E-05*t**2 - 0.0384*t + 17.94
            !IF (t.LT.369.396)THEN
            !    fkf=8.0001
            !ELSEIF (t.GT.1921.7) THEN
            !    fkf=2.2898
            !ENDIF
            RETURN
      END FUNCTION fkf

      FUNCTION fkc(t)
      
            IMPLICIT NONE
      
            ! 4-th polynomial for thermal conductivity of Zr in w/m-C, t in K
            INTEGER(sik) :: n
            REAL(sdk) :: fkc,t

            IF (num_kclad .GT. 0) THEN
                fkc = 0.0
                DO n=1,num_kclad
                    fkc = fkc + dat_kclad(n) * t ** (n-1)
                END DO
            ELSE            
                fkc = 0.0000000000030024*t**4-0.00000001027*t**3+0.000018274*t**2-0.004965*t+11.839
            END IF
            !IF (t<473.15)THEN
            !fkc=12.0046
            !ELSEIF (t>2000.00) THEN
            !fkc=1.0E10
            !ENDIF
            RETURN
      END FUNCTION fkc

END MODULE PATHS_FuelProbM
