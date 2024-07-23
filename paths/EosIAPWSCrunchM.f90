MODULE EosIAPWSCrunch
!
      USE IntrType, ONLY: sdk, sik
!
      IMPLICIT NONE
!     
      INTEGER(sik), PARAMETER, PRIVATE :: sdkx = sdk
!
!  This module contains the IAPWS steam table routines (same that RELAP5 uses)
!   Added to TracM in July 2000 by Glen Mortensen, Rex Shumway,
!   Doug Barber, and Randy Tompot (the Idaho crew of ISL)
!
CONTAINS

      SUBROUTINE getpsat(tables, t, s, err)
!
!  Returns the saturation pressure given the temperature
!
!  Cognizant engineer:  rwt,gam
!
         USE EosIAPWSData, ONLY: psatbar, ttrip, ofirstsattemp, olastsattemp, ptable1, ptable3, ntable3, lastTSatInd
         USE Io, ONLY: iout
!
         IMPLICIT NONE
!
         REAL(sdk) :: delt, s(:), tleft, tright, t, tables(:), z
         INTEGER(sik) :: g, itleft, itleftlimit, itright, itrightlimit, opl, opsattleft
         LOGICAL :: err
!
!      call timstart('getpsat')
!
         err = .FALSE.
!
         itleftlimit = ofirstsattemp
         itrightlimit = olastsattemp
         t = MAX(ttrip, t)
!
         CALL getindex(tables, t, ptable1, itleftlimit, itrightlimit, itleft, err, lastTSatInd)
!
!rex+ 15 Mar 2001
         IF (err) THEN
            WRITE (iout, 2360) t
2360        FORMAT ('  ***GETDSAT err, t=', e14.6)
            WRITE(*,*) 'Fatal Error' !CALL error(4, '*getpsat* error getting index from tables')
         END IF
!rex-
!
         itright = itleft + 1
!
         tleft = tables(ptable1+itleft)
         tright = tables(ptable1+itright)
!
         opsattleft = ptable3 + (itleft-itleftlimit) * ntable3
!
         opl = opsattleft + 1
!
         delt = tright - tleft
!         delt2 = (tright-tleft) ** 2
         z = (t-tleft) / (tright-tleft)
         g = opl + 2
         s(psatbar) =    tables(g+1)      &
          &         + z*(tables(g+2)      &
          &         + z*(tables(g+3)      &
          &         + z*(tables(g+4)      &
          &         + z*(tables(g+5)      &
          &         + z* tables(g+6)))))
         s(psatbar) = EXP(s(psatbar))
!
         RETURN
      END SUBROUTINE getpsat
!     
      SUBROUTINE gettsat(tables, p, s, err)
!
!  Returns the saturation temperature given pressure.
!  This routine is similiar to getsatatp except that it only
!  returns the saturation temperature.
!
!  Cognizant engineer:  rwt,gam
!
         USE EosIAPWSData, ONLY: ofirstsatpres, olastsatpres, ptable2, ptable4, ntable4, tsatbar, lastPSatInd
         USE Io, ONLY: iout
!
         IMPLICIT NONE
!
         REAL(sdk) :: delt, p, phi, plo, s(:), t, tables(:), z
         INTEGER(sik) :: iphi, iplo, ofirstpres, olastpres, otsatplo, otsatphi
         LOGICAL :: err
!
!      call timstart('gettsat')
!
         err = .FALSE.
!
         ofirstpres = ofirstsatpres
         olastpres = olastsatpres
!
         CALL getindex(tables, p, ptable2, ofirstpres, olastpres, iplo, err, lastPSatInd)
!
!rex+ 15 Mar 2001
         IF (err) THEN
            WRITE (iout, 2360) p
2360        FORMAT ('  ***GETTSAT err, p=', e14.6)
            WRITE (*,*) 'Fatal Error' !CALL error(4, '*gettsat* Error getting index from tables')
         END IF
!rex-
!
         iphi = iplo + 1
         plo = tables(ptable2+iplo)
         phi = tables(ptable2+iphi)
!
         otsatplo = ptable4 + (iplo-ofirstpres) * ntable4
         otsatphi = ptable4 + (iphi-ofirstpres) * ntable4
!
!  compute saturation temperature using the 2-point formula
!
         delt = LOG(phi) - LOG(plo)
         z = (LOG(p)-LOG(plo)) / delt
         t =    tables(otsatplo+4)   &
       &   + z*(tables(otsatplo+5)   &
       &   + z*(tables(otsatplo+6)   &
       &   + z*(tables(otsatplo+7)   &
       &   + z*(tables(otsatplo+8)   &
       &   + z* tables(otsatplo+9)))))
!
         s(tsatbar) = t
!
         RETURN
      END SUBROUTINE gettsat            
!
      SUBROUTINE getsatatp(tables, p, s, err)
!
!  Returns the saturation properties specified saturation
!  pressure. This routine replaces sth2x0.F and sth2x1.F
!  and sth2x2.F
!
!  Cognizant engineer:  rwt
!
         USE EosIAPWSData, ONLY: ttrip, ofirstsatpres, olastsatpres, ptable2, ptable4, ntable4, lastPSatInd
         USE EosIAPWSData, ONLY: tempbar, tsatbar, hsubf, hsubg, usubf, usubg, vsubf, vsubg
         USE Io, ONLY: iout
!
         IMPLICIT NONE
!
         REAL(sdk) :: delt, p, phi, plo, s(:), t, tables(:), z
         INTEGER(sik) :: iphi, iplo, ofirstpres, olastpres, otsatplo, otsatphi
         LOGICAL :: err
!
!      call timstart('getsatatp')
!
         err = .FALSE.
!
         ofirstpres = ofirstsatpres
         olastpres = olastsatpres
!
         CALL getindex(tables, p, ptable2, ofirstpres, olastpres, iplo, err, lastPSatInd)
!
!rex+ 15 Mar 2001
         IF (err) THEN
            WRITE (iout, 2360) p
2360        FORMAT ('  ***GETSATATP err, p=', e14.6)
            !CALL error(4, '*getsatatp* error getting index from tables')
         END IF
!rex-
!
         iphi = iplo + 1
         plo = tables(ptable2+iplo)
         phi = tables(ptable2+iphi)
!
         otsatplo = ptable4 + (iplo-ofirstpres) * ntable4
         otsatphi = ptable4 + (iphi-ofirstpres) * ntable4
!
!  Load a array for 2-point formula
!
         delt = LOG(phi) - LOG(plo)
         z = (LOG(p)-LOG(plo)) / delt
         t =    tables(otsatplo+4)   &
       &   + z*(tables(otsatplo+5)   &
       &   + z*(tables(otsatplo+6)   &
       &   + z*(tables(otsatplo+7)   &
       &   + z*(tables(otsatplo+8)   &
       &   + z* tables(otsatplo+9)))))
!
         s(tempbar) = t
         t = MAX(ttrip, t)
!
         CALL getsatatt(tables, t, s, err)
!
         s(tsatbar) = t
         s(hsubf) = s(usubf) + p * s(vsubf)
         s(hsubg) = s(usubg) + p * s(vsubg)
!
         RETURN
      END SUBROUTINE getsatatp
!
      SUBROUTINE getsatatt(tables, t, s, err)
!
!  Returns the saturation properties specified saturation
!  temperature. This routine replaces sth2x0.F and sth2x1.F
!  and sth2x2.F
!
!  Cognizant engineer:  rwt
!
         USE Io, ONLY: iout
         USE EosIAPWSData, ONLY: ttrip, ofirstsattemp, olastsattemp, ptable1, ptable3, ntable3, psatbar
         USE EosIAPWSData, ONLY: vsubf, usubf, betasubf, kapasubf, cpsubf, entpysubf, hsubf, lastTSatInd
         USE EosIAPWSData, ONLY: vsubg, usubg, betasubg, kapasubg, cpsubg, entpysubg, hsubg
!
         IMPLICIT NONE
!
         REAL(sdk) :: a(2), delt, s(:), tleft, tright, t, tables(:)
         REAL(sdk) :: xy(2), y(12), z
         INTEGER(sik) :: g, i, itleft, itleftlimit, itright, itrightlimit, ol, opl, opr, opsattleft, &
        & opsattright, or
         LOGICAL :: err
!
!      call timstart('getsatatt')
!
         err = .FALSE.
!
         itleftlimit = ofirstsattemp
         itrightlimit = olastsattemp
         t = MAX(ttrip, t)
         CALL getindex(tables, t, ptable1, itleftlimit, itrightlimit, itleft, err, lastTSatInd)
!
         IF (err) THEN
            WRITE (iout, 2660) t
2660        FORMAT (' ***GETSATATT, err, t=', es14.3)
            !CALL error(1, '*getsatatt* error getting index from tables')
         END IF
!
         itright = itleft + 1
!
         tleft = tables(ptable1+itleft)
         tright = tables(ptable1+itright)
!
         opsattleft = ptable3 + (itleft-itleftlimit) * ntable3
         opsattright = ptable3 + (itright-itleftlimit) * ntable3
!
!  Load a, b, and c arrays for 2-point formula
!
         a(1) = tleft
         a(2) = tright
!
         i = 1
         opl = opsattleft + 9 * (i-1) + 1
         opr = opsattright + 9 * (i-1) + 1
         delt = tright - tleft
!         delt2 = (tright-tleft) ** 2
         z = (t-tleft) / (tright-tleft)
         g = opl + 2
         s(psatbar) =    tables(g+1)      &
       &            + z*(tables(g+2)      &
       &            + z*(tables(g+3)      &
       &            + z*(tables(g+4)      &
       &            + z*(tables(g+5)      &
       &            + z* tables(g+6)))))
         s(psatbar) = EXP(s(psatbar))
!
         i = 2
         ol = opsattleft + 9 * (i-1) + 1
         or = opsattright + 9 * (i-1) + 1
!
         xy(1) = tleft
         xy(2) = tright
!
         CALL satprops(xy, ol, or, tables, t, y, err)
!
         s(vsubf) = y(1)
         s(usubf) = y(2)
         s(betasubf) = y(3)
         s(kapasubf) = y(4)
         s(cpsubf) = y(5)
         s(entpysubf) = y(6)
!
         s(vsubg) = y(7)
         s(usubg) = y(8)
         s(betasubg) = y(9)
         s(kapasubg) = y(10)
         s(cpsubg) = y(11)
         s(entpysubg) = y(12)
!
!  Compute enthalpy saturation values
!
         s(hsubf) = s(usubf) + s(psatbar) * s(vsubf)
         s(hsubg) = s(usubg) + s(psatbar) * s(vsubg)
!
!  Compute the mixture properties - TRACE doesn't use mixture properties.
!  So the following line was commented out.
!
!         CALL mixprops(s, getprops)
!
         RETURN
      END SUBROUTINE getsatatt
!
      SUBROUTINE satprops(xy, p1, p2, tables, x, y, err)
!
!  Evaluates the saturation properties using Hermite polonomials
!
!  Cognizant engineer:  rwt 5/25/1999
!  modification of herm1d; gam 1/25/1000
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Local variables
!
         INTEGER(sik) :: g
         REAL(sdk) :: xy(:), tables(:), x, y(:)
         REAL(sdk) :: x1
         REAL(sdk) :: delt
         INTEGER(sik) :: p1, p2, p1p1, p2p1, p1p2, p2p2
         LOGICAL :: err
         INTEGER(sik) :: nprop, i
         DATA nprop / 12 /
!
         err = .FALSE.
         delt = xy(2) - xy(1)
!         delt2 = delt**2
         x1 = (x-xy(1)) / delt
!
         DO 10 i = 1, nprop
            p1p1 = p1 + 1
            p2p1 = p2 + 1
            p1p2 = p1 + 2
            p2p2 = p2 + 2
!
            g = p1 + 2
            y(i) =     tables(g+1) &
        &        + x1*(tables(g+2) &
        &        + x1*(tables(g+3) &
        &        + x1*(tables(g+4) &
        &        + x1*(tables(g+5) &
        &        + x1*tables(g+6)))))
!  bump the pointers to the next guy in the table
!
            p1 = p1 + 9
            p2 = p2 + 9
10       CONTINUE
!
         RETURN
      END SUBROUTINE satprops
!      
      SUBROUTINE getsnglatpt(tables, fluidState, getprops, t, p, s, err, iterate)
!
!  Returns the single phase properties for subcritical liquid or
!  vapor or for a supercritical fluid at a specified(p,t) point.
!  This routine replaces sth2x3.F
!
!  Cognizant engineer:  rwt
!
         USE EosIAPWSData, ONLY: tcrit, liquid, vapor, supercritical, ptable2, ptable1, lastTInd, lastPInd
         USE EosIAPWSData, ONLY: ubar, vbar, hbar, entpybar, kapabar, betabar, cpbar, qualbar, tsatbar, tempbar
         USE EosIAPWSData, ONLY: vsubf, usubf, hsubf, betasubf, kapasubf, cpsubf, entpysubf
         USE EosIAPWSData, ONLY: vsubg, usubg, hsubg, betasubg, kapasubg, cpsubg, entpysubg, g
         USE Io, ONLY: iout
!
         IMPLICIT NONE
!
         REAL(sdk) :: a(11, 4), deltt, p, plo, phi, s(:), t, tables(:), tleft, tleftlimitplo, &
        & tleftlimitphi, tright, trightlimitplo, trightlimitphi
         INTEGER(sik) :: iplo, iphi, otleftlimit, otrightlimit, ptableprop, ntablelimits, ptablelimits, &
        & ofirstpres, olastpres, itleftlimitplo, itleftplo, itrightplo, itrightlimitplo, stableprop, offsetu, &
        & omint, omaxt, ntableprop, itleftlimitphi, itleftphi, itrightphi, itrightlimitphi, itleft, itright, &
        & ptr
         INTEGER(sik), SAVE :: lastIndex
!
         LOGICAL :: err, getprops(:), threepointleft, iterate
         INTEGER(sik), INTENT(IN) :: fluidState
         PARAMETER(offsetu=1)
         PARAMETER(omint=1)
         PARAMETER(omaxt=2)
         PARAMETER(otleftlimit=5)
         PARAMETER(otrightlimit=6)
!
         CHARACTER(LEN=120) :: mess
!
         err = .FALSE.
         threepointleft = .FALSE.
         iterate = .FALSE.
!
!  Find the table and set generic pointers
!
         CALL setpointers(fluidState, ofirstpres, olastpres, ptableprop, stableprop, ntableprop, ptablelimits, &
        & ntablelimits, lastIndex)
!
!  Find plo, phi, iplo, and iphi
!
         CALL getindex(tables, p, ptable2, ofirstpres, olastpres, iplo, err, lastPInd)
!
!rex+ 15 Mar 2001
         IF (err) THEN
            WRITE (iout, 2360) p, t
2360        FORMAT ('  ***GETSNGLATPT err, p=', e14.6, ', t=', e14.6)
            !CALL error(2, '*getsnglatpt* Error getting index from tables')
         ELSE
!rex-
!
!   Load bounding pressure indices
!
            iphi = iplo + 1
            plo = tables(ptable2+iplo)
            phi = tables(ptable2+iphi)
!
!  Find bounding t indices on plo line from Table 8, 9, or 10
!
            tleftlimitplo   = tables(ptablelimits &
           &                 + (iplo - ofirstpres)*ntablelimits + omint)
            trightlimitplo  = tables(ptablelimits &
           &                 + (iplo - ofirstpres)*ntablelimits + omaxt)
            itleftlimitplo  = int(tables(ptablelimits &
           &                 + (iplo - ofirstpres)*ntablelimits &
           &                 + otleftlimit))
            itrightlimitplo = int(tables(ptablelimits &
           &                 + (iplo - ofirstpres)*ntablelimits &
           &                 + otrightlimit))
!
            IF (t .LT. tleftlimitplo) THEN
               itleftplo = itleftlimitplo
               itrightplo = itleftplo + 1
            ELSE IF (t .GT. trightlimitplo) THEN
               itleftplo = itrightlimitplo - 1
               itrightplo = itleftplo + 1
            ELSE
               CALL getindex(tables, t, ptable1, itleftlimitplo, itrightlimitplo, itleftplo, err, lastTInd)
               itrightplo = itleftplo + 1_sik
            END IF
!
!           Check for error from getindex.
            IF (err) THEN
               WRITE (mess, '(a, es14.4, a)') ' *getsnglatpt* t = ', t, ' table plo index error.'
               !CALL error(2, mess)
            ELSE
!
!  Find bounding t indices on phi line from Table 8, 9, or 10
!
               tleftlimitphi  = tables(ptablelimits &
              &                + (iphi - ofirstpres)*ntablelimits + omint)
               trightlimitphi = tables(ptablelimits &
              &                 + (iphi - ofirstpres)*ntablelimits + omaxt)
               itleftlimitphi  = int(tables(ptablelimits &
              &                + (iphi - ofirstpres)*ntablelimits &
              &                + otleftlimit))
               itrightlimitphi = int(tables(ptablelimits &
              &                 + (iphi - ofirstpres)*ntablelimits &
              &                 + otrightlimit))
!
               IF (t .LT. tleftlimitphi) THEN
                  itleftphi = itleftlimitphi
                  itrightphi = itleftphi + 1
               ELSE IF (t .GT. trightlimitphi) THEN
                  itleftphi = itrightlimitphi - 1
                  itrightphi = itleftphi + 1
               ELSE
                  CALL getindex(tables, t, ptable1, itleftlimitphi, itrightlimitphi, itleftphi, err, lastTInd)
                  itrightphi = itleftphi + 1_sik
               END IF
               IF (err) THEN
                  WRITE (mess, '(a, es14.4, a)') ' *getsnglatpt* t = ', t, ' table phi index error.'
                  !CALL error(2, mess)
               ELSE
!
!  Move temp indices to make a rectangle if a rhombus
!
                  call setanchor (itleftphi,itleftplo,itrightplo,itrightphi,      &
                 &                 itleftlimitplo,itrightlimitplo,itleftlimitphi,  &
                 &                 itrightlimitphi,p,plo,phi,itleft,itright,       &
                 &                 threepointleft,err)
!
                  IF (err) THEN
                     WRITE (mess, '(3(a, es14.4))') ' *getsnglatpt* error from setanchor p = ', p, &
                    & ' plo = ', plo, ' phi = ', phi
                     !CALL error(2, mess)
                  ELSE
!
                     tleft = tables(ptable1+itleft)
                     tright = tables(ptable1+itright)
                     deltt = tright - tleft
!
!  Calculate t at the endpoints of the input p line
!  using the 4-point formula
!
                     CALL loadcorners(tables, a, itleft, itright, iplo, ptableprop, stableprop, ntableprop, &
                    & ofirstpres, ptr)
!
!  Although err is passed back from the following routines, none of the following
!  routines include logic to set err to .TRUE.  If at a future date, any of these
!  routines include logic that can set err to .TRUE., then logic should be added
!  in this routine to write an appropriate message and skip to the end of this routine.
!    herm2d, RhoLine, Kappaline, BetaLine, Beta2d, and Cpline.
                     CALL herm2d(tables, ptr, a, fluidState, g, t, p, err)
!
!  Calculate single-phase properties using the returned
!  elements of g
                     IF (g(2) < 0.0_sdk) THEN
                        CALL Rholine(tables, ptr, a, fluidState, g, t, p, err)
                     END IF
                     IF (g(3) .GT. 0.0_sdk) THEN
                        CALL Kappaline(tables, ptr, a, fluidState, g, t, p, err)
                     END IF
                     IF (g(5) .LT. 0.0_sdk) THEN
                        CALL Betaline(tables, ptr, a, fluidState, g, t, p, err)
                        IF (g(5) .LT. 0.0_sdk) THEN
                           IF (t .GT. tright) THEN
                              CALL Betaline(tables, ptr, a, fluidState, g, tright, p, err)
                           ELSE IF (t .LT. tleft) THEN
                              CALL Betaline(tables, ptr, a, fluidState, g, tleft, p, err)
                           END IF
                        END IF
                        IF (g(5) .LT. 0.0_sdk) THEN
                           CALL Beta2d(tables, ptr, a, fluidState, g, t, p, err)
                        END IF
                        IF (g(5) .LT. 0.0_sdk) THEN
                           CALL Beta2d(tables, ptr, a, fluidState, g, tright, phi, err)
                        END IF
                     END IF
                     IF (g(7) .GT. 0.0_sdk) THEN
                        CALL Cpline(tables, ptr, a, fluidState, g, t, p, err)
                     END IF
!
                     IF (getprops(ubar)) THEN
                        s(ubar) = g(1) - t * g(4) - p * g(2)
                     END IF
                     IF (getprops(vbar)) THEN
                        s(vbar) = g(2)
                     END IF
                     IF (getprops(entpybar)) THEN
                        s(entpybar) = - g(4)
                     END IF
                     IF (getprops(kapabar)) THEN
                        s(kapabar) = - g(3) / g(2)
                     END IF
                     IF (getprops(betabar)) THEN
                        s(betabar) = g(5) / g(2)
                     END IF
                     IF (getprops(cpbar)) THEN
                        s(cpbar) = - t * g(7)
                     END IF
                     IF (getprops(hbar)) THEN
                        s(hbar) = g(1) - t * g(4)
                     END IF
                     IF (getprops(qualbar)) THEN
                        SELECT CASE(fluidState)
                        CASE (liquid)
                           s(qualbar) = 0.0_sdk
                        CASE (vapor)
                           s(qualbar) = 1.0_sdk
                        CASE (supercritical)
!gam  set sat properties equal to single phase properties
                           s(tsatbar)   = s(tempbar)
                           s(vsubf)  = s(vbar)
                           s(vsubg)  = s(vbar)
                           s(usubf)  = s(ubar)
                           s(usubg)  = s(ubar)
                           s(hsubf)  = s(hbar)
                           s(hsubg)  = s(hbar)
                           s(betasubf)  = s(betabar)
                           s(betasubg)  = s(betabar)
                           s(kapasubf)  = s(kapabar)
                           s(kapasubg)  = s(kapabar)
                           s(cpsubf)    = s(cpbar)
                           s(cpsubg)    = s(cpbar)
                           s(entpysubf) = s(entpybar)
                           s(entpysubg) = s(entpybar)
                           s(qualbar)   = 0.0_sdk
                        END SELECT
                     END IF
!
!  Check interpolated values
!
                     CALL checkvalue(fluidState, p, g, s, getprops, err, iterate)
                     IF (err) THEN
                        WRITE (mess, '(2(a, es14.4))') ' *getsnglatpt* error from checkvalue p = ', p, &
                       & ' t = ', s(tempbar)
                        !CALL error(2, mess)
                     END IF
!
                  END IF
               END IF
            END IF
         END IF
!
         RETURN
      END SUBROUTINE getsnglatpt
!
      SUBROUTINE getsnglatpu(tables, fluidState, getprops, u, p, s, err, iterate)
!
!  Returns the single phase properties for subcritical liquid or
!  vapor or for a supercritical fluid at a specified(p,u) point.
!  This routine replaces sth2x6.F and sth2xf.F
!
!  Cognizant engineer:  rwt
!
         USE Io, ONLY: iout
         USE EosIAPWSData, ONLY: tcrit, vapor, liquid, supercritical, ptable2, ptable1, ubar, lastPInd
         USE EosIAPWSData, ONLY: tempbar, vbar, entpybar, kapabar, betabar, cpbar, hbar, qualbar, tsatbar
         USE EosIAPWSData, ONLY: vsubf, usubf, hsubf, betasubf, kapasubf, cpsubf, entpysubf
         USE EosIAPWSData, ONLY: vsubg, usubg, hsubg, betasubg, kapasubg, cpsubg, entpysubg, g
!
         IMPLICIT NONE
!
         REAL(sdk) :: a(11, 4), b(8), deltp, deltt, p, s(:), tables(:), tint, tleft, tright, tuatleft, &
        & tuatright, u, uatleft, uatright, utatleft, utatright, plo, phi, uleftlimitplo, uleftlimitphi, &
        & urightlimitplo, urightlimitphi
         REAL(sdk) :: tintm5, tintp5, dtint, tintsave, uatleftsave
!
         INTEGER(sik) :: i, iplo, iphi, otleftlimit, otrightlimit, ptableprop, ntablelimits, ptablelimits, &
        & ofirstpres, olastpres, uindex, itleftlimitplo, itleftplo, itrightplo, itrightlimitplo, &
        & itleftlimitphi, itleftphi, itrightphi, itrightlimitphi, itleft, itright, offsetu, ominu, omaxu, &
        & ntableprop, ptr, stableprop
         INTEGER(sik), SAVE :: lastIndex
!
         LOGICAL :: err, getprops(26), leftshift, rightshift, threepointleft, iterate
!rex+ 20 Mar 2001
         REAL(sdk) gtop(11), y
!rex-
         INTEGER(sik), INTENT(IN) :: fluidState
         PARAMETER(offsetu=1)
         PARAMETER(ominu=3)
         PARAMETER(omaxu=4)
         PARAMETER(otleftlimit=5)
         PARAMETER(otrightlimit=6)
         PARAMETER(dtint=2.0_sdk)
!
         err = .FALSE.
         leftshift = .FALSE.
         rightshift = .FALSE.
         threepointleft = .FALSE.
!
!  Find the table and set generic pointers
!
          call setpointers (fluidState,ofirstpres,olastpres,ptableprop, &
        &                   stableprop,ntableprop,ptablelimits,  &
        &                   ntablelimits, lastIndex)
!
!  Find plo, phi, iplo, and iphi
!
         CALL getindex(tables, p, ptable2, ofirstpres, olastpres, iplo, err, lastPInd)
!
!rex+ 15 Mar 2001
         IF (err) THEN
            WRITE (iout, 2360) p, u
2360        FORMAT ('  ***GETSNGLATPU err, p=', e14.6, ', u=', e14.6)
            !CALL error(4, '*getsnglatpu* Error getting index from tables')
         END IF
!rex-
!
!   Load bounding pressure indices
!
         iphi = iplo + 1
         plo = tables(ptable2+iplo)
         phi = tables(ptable2+iphi)
         deltp = phi - plo
!
!  Find bounding u indices on plo line from Table 8, 9, or 10
!
         uleftlimitplo   = tables(ptablelimits &
       &                 + (iplo - ofirstpres)*ntablelimits + ominu)
         urightlimitplo  = tables(ptablelimits &
       &                 + (iplo - ofirstpres)*ntablelimits + omaxu)
         itleftlimitplo  = int(tables(ptablelimits &
       &                 + (iplo - ofirstpres)*ntablelimits &
       &                 + otleftlimit))
         itrightlimitplo = int(tables(ptablelimits &
       &                 + (iplo - ofirstpres)*ntablelimits &
       &                 + otrightlimit))
!
         IF (u .LT. uleftlimitplo) THEN
            itleftplo = itleftlimitplo
            itrightplo = itleftplo + 1
         ELSE IF (u .GT. urightlimitplo) THEN
            itleftplo = itrightlimitplo - 1
            itrightplo = itleftplo + 1
         ELSE
            DO 10 i = itleftlimitplo + 1, itrightlimitplo
               uindex = ptableprop  &
         &            + (iplo - ofirstpres)*stableprop &
         &            + (i - 1)*ntableprop &
         &            + offsetu
               IF (tables(uindex) .GE. u) THEN
                  itleftplo = i - 1
                  itrightplo = i
                  GO TO 20
               END IF
10          CONTINUE
            err = .TRUE.
            GO TO 999
20          CONTINUE
         END IF
!
!  Find bounding u indices on phi line from Table 8, 9, or 10
!
         uleftlimitphi   = tables(ptablelimits &
       &                 + (iphi - ofirstpres)*ntablelimits + ominu)
         urightlimitphi  = tables(ptablelimits &
       &                 + (iphi - ofirstpres)*ntablelimits + omaxu)
         itleftlimitphi  = int(tables(ptablelimits &
       &                 + (iphi - ofirstpres)*ntablelimits &
       &                 + otleftlimit))
         itrightlimitphi = int(tables(ptablelimits &
       &                 + (iphi - ofirstpres)*ntablelimits &
       &                 + otrightlimit))
!
         IF (u .LT. uleftlimitphi) THEN
            itleftphi = itleftlimitphi
            itrightphi = itleftphi + 1
         ELSE IF (u .GT. urightlimitphi) THEN
            itleftphi = itrightlimitphi - 1
            itrightphi = itleftphi + 1
         ELSE
            DO 30 i = itleftlimitphi + 1, itrightlimitphi
               uindex = ptableprop &
         &            + (iphi - ofirstpres)*stableprop &
         &            + (i - 1)*ntableprop &
         &            + offsetu
               IF (tables(uindex) .GE. u) THEN
                  itleftphi = i - 1
                  itrightphi = i
                  GO TO 40
               END IF
30          CONTINUE
            err = .TRUE.
            GO TO 999
40          CONTINUE
         END IF
!
!  Move temp indices to make a rectangle if a rhombus
!
         CALL setanchor(itleftphi, itleftplo, itrightplo, itrightphi, itleftlimitplo, itrightlimitplo, &
        & itleftlimitphi, itrightlimitphi, p, plo, phi, itleft, itright, threepointleft, err)
!
         IF (err) GO TO 999
!
!
         leftshift = .FALSE.
         rightshift = .FALSE.
!
888      tleft = tables(ptable1+itleft)
         tright = tables(ptable1+itright)
         deltt = tright - tleft
!
!  Calculate u at the endpoints of the input p line
!  using the 4-point formula
!
         CALL loadcorners(tables, a, itleft, itright, iplo, ptableprop, stableprop, ntableprop, ofirstpres, &
        & ptr)
!
!rex+ 9 Feb 2001; if high press and threepoint or liquid
         IF (p .GT. 21.70d6 .AND. threepointleft) THEN
!
            CALL herm2d(tables, ptr, a, fluidState, gtop, tright, phi, err)
!
            y = (p-plo) / deltp
!
            CALL herm2d(tables, ptr, a, fluidState, g, tleft, plo, err)
!
            g(1) = g(1) + (gtop(1)-g(1)) * y
            g(2) = g(2) + (gtop(2)-g(2)) * y
            g(3) = g(3) + (gtop(3)-g(3)) * y
            g(4) = g(4) + (gtop(4)-g(4)) * y
            g(5) = g(5) + (gtop(5)-g(5)) * y
            g(7) = g(7) + (gtop(7)-g(7)) * y
!
            tint = tleft + (tright-tleft) * y
!
            GO TO 1020
         END IF
!rex-
!
!  Recalculate left hand side temperature if we have a triangle
!
!rex+ 28 Mar 2001 add another restriction on next if test
         IF (threepointleft) THEN
            tleft = tleft + (tright-tleft) * (p-plo) / (phi-plo)
         END IF
!rex-
!
!
!  Interpolate for the value of u at left point
!
         CALL herm2dplus(tables, ptr, a, fluidState, g, tleft, p, err)
!
!  Calculate single-phase liquid properties using the elements of g
!
         uatleft = g(1) - tleft * g(4) - p * g(2)
         utatleft = - tleft * g(7) - p * g(5)
         tuatleft = 1.0_sdk / utatleft
!
!  Interpolate for the value of u at right point
!
         CALL herm2dplus(tables, ptr, a, fluidState, g, tright, p, err)
!
!  Calculate single-phase liquid properties using the elements of g
!
         uatright = g(1) - tright * g(4) - p * g(2)
         utatright = - tright * g(7) - p * g(5)
         tuatright = 1.0_sdk / utatright
!
!  Use the square we have been working in?
!  if at the leftlimit, do not shift over.
!
!rex+ 23 Mar 2001 do not shift if not threepointleft
         IF ((u .LT. uatleft) .AND. ( .NOT. rightshift) .AND. ( .NOT. threepointleft)) THEN
!rex-
            itleft = itleft - 1
            itright = itleft + 1
            leftshift = .TRUE.
            IF (itleft .LT. itleftlimitplo) THEN
               itleft = itleft + 1
               itright = itleft + 1
               GO TO 777
            ELSE IF (itleft .LT. itleftlimitphi) THEN
               threepointleft = .TRUE.
               itright = itleftlimitphi
            END IF
            GO TO 888
         ELSE IF ((u .GT. uatright) .AND. ( .NOT. leftshift)) THEN
            itleft = itleft + 1
            itright = itleft + 1
            rightshift = .TRUE.
            IF ((itright .GT. itrightlimitplo) .OR. (itright .GT. itrightlimitphi)) THEN
               itleft = itleft - 1
               itright = itleft + 1
               GO TO 777
            END IF
            IF (threepointleft) THEN
               itright = itleftlimitphi
               IF (itleft .EQ. itleftlimitphi) THEN
                  itright = itleft + 1
                  threepointleft = .FALSE.
               END IF
            END IF
            GO TO 888
         END IF
!
777      b(1) = uatleft
         b(2) = tleft
         b(3) = tuatleft
!      b(4) = tuuatleft
         b(5) = uatright
         b(6) = tright
         b(7) = tuatright
!
         CALL cubic(b, u, tint, err)
!
         IF ((u .GT. uatright) .AND. (tint .LT. tright)) THEN
!
!rex+ 1 Mar 2001 TDV for case pcrit.i above 21 MPa had error
!       but timestep reductions do not occur from tstate
!       plus answers were fine if it continued.
            IF (p .GT. 20.0d6) THEN
               tint = tright
               err = .false.
               IF (tint .le. 0.0d0) THEN
                 err = .true.
                 RETURN
               END IF
!
               CALL herm2d(tables, ptr, a, fluidState, g, tint, p, err)
!
               GO TO 1020
            ELSE
               err = .TRUE.
               GO TO 999
            END IF
!rex-
         END IF
!gam  use linear extrapolation instead of cubic fit if cubic fails
         IF (u .LT. uatleft .AND. tint .GT. tleft) THEN
            tintsave = tint
            tint = tleft + (u-uatleft) * (tright-tleft) / (uatright-uatleft)
         END IF
!
!   Calculate single-phase properties using interpolated temperature and
!   the same 4 corner points already loaded into the a array
!
779      CONTINUE
!rex+ 8 Mar 2001
         IF (tint .LT. tleft .AND. u .GT. uatleft) THEN
            tint = tleft + (u-uatleft) / (uatright-uatleft) * (tright-tleft)
         END IF
         IF (tint .GT. tright .AND. u .LT. uatright) THEN
            tint = tleft + (u-uatleft) / (uatright-uatleft) * (tright-tleft)
         END IF
!rex-
         IF (tint .LT. tleft - deltt) THEN
!gam  get a better estimate for tint
            uatleftsave = uatleft
            IF (tint <= 0.0_sdk) THEN
               err = .TRUE.
               RETURN
            END IF
!
            CALL herm2dleft(tables, ptr, a, fluidState, g, tint, p, err)
!
            uatleft = g(1) - tint * g(4) - p * g(2)
            utatleft = - tint * g(7) - p * g(5)
            tuatleft = 1.0_sdk / utatleft
!  now do a linear interpolation between this value and the tleft value
            tint = tint + (u-uatleft) * (tleft-tint) / (uatleftsave-uatleft)
!  now home in on it by getting u at points tint+dtint and tint-dtint
!rex+ 27 Mar 2001 dtint to big, use deltt
            tintm5 = tint - deltt
!rex-
            IF (tintm5 <= 0.0_sdk) THEN
               err = .TRUE.
               RETURN
            END IF
!
            CALL herm2dleft(tables, ptr, a, fluidState, g, tintm5, p, err)
!
            uatleft = g(1) - tintm5 * g(4) - p * g(2)
            utatleft = - tintm5 * g(7) - p * g(5)
            tuatleft = 1.0_sdk / utatleft
!
!rex+ 27 Mar 2001 dtint to big, use deltt
            tintp5 = tint + deltt
!rex-
            IF (tintp5 <= 0.0_sdk) THEN
               err = .TRUE.
               RETURN
            END IF
!
            CALL herm2dleft(tables, ptr, a, fluidState, g, tintp5, p, err)
!
            uatright = g(1) - tintp5 * g(4) - p * g(2)
            utatright = - tintp5 * g(7) - p * g(5)
            tuatright = 1.0_sdk / utatright
!
            b(1) = uatleft
            b(2) = tintm5
            b(3) = tuatleft
            b(5) = uatright
            b(6) = tintp5
            b(7) = tuatright
            CALL cubic(b, u, tint, err)
!gam  now get the properties at this temperature
            IF (tint <= 0.0_sdk) THEN
               err = .TRUE.
               RETURN
            END IF

            CALL herm2dleft(tables, ptr, a, fluidState, g, tint, p, err)
!
         ELSE
            IF (tint <= 0.0_sdk) THEN
               err = .TRUE.
               RETURN
            END IF
!
            CALL herm2d(tables, ptr, a, fluidState, g, tint, p, err)
!
         END IF
!
!rex+ 9 Feb 2001
1020     CONTINUE
         IF (g(2) < 0.0_sdk) THEN
!
            CALL Rholine(tables, ptr, a, fluidState, g, tint, p, err)
!
         END IF
         IF (g(3) .GT. 0.0_sdk) THEN
!
            CALL Kappaline(tables, ptr, a, fluidState, g, tint, p, err)
!
         END IF
         IF (g(5) .LT. 0.0_sdk) THEN
!
            CALL Betaline(tables, ptr, a, fluidState, g, tint, p, err)
!
            IF (g(5) .LT. 0.0_sdk) THEN
               IF (tint .GT. tright) THEN
!
                  CALL Betaline(tables, ptr, a, fluidState, g, tright, p, err)
!
               ELSE IF (tint .LT. tleft) THEN
!
                  CALL Betaline(tables, ptr, a, fluidState, g, tleft, p, err)
!
               END IF
            END IF
            IF (g(5) .LT. 0.0_sdk) THEN
!
               CALL Beta2d(tables, ptr, a, fluidState, g, tint, p, err)
!
            END IF
            IF (g(5) .LT. 0.0_sdk) THEN
!
               CALL Beta2d(tables, ptr, a, fluidState, g, tright, phi, err)
!
            END IF
         END IF
         IF (g(7) .GT. 0.0_sdk) THEN
!
            CALL Cpline(tables, ptr, a, fluidState, g, tint, p, err)
!
         END IF
!rex-
!  Calculate single-phase liquid properties using the returned
!  elements of g
!
         IF (getprops(tempbar)) THEN
            s(tempbar) = tint
         END IF
         IF (getprops(vbar)) THEN
            s(vbar) = g(2)
         END IF
         IF (getprops(entpybar)) THEN
            s(entpybar) = - g(4)
         END IF
         IF (getprops(kapabar)) THEN
            s(kapabar) = - g(3) / g(2)
         END IF
         IF (getprops(betabar)) THEN
            s(betabar) = g(5) / g(2)
            s(betabar)  = MIN(s(betabar),0.05_sdk)
         END IF
         IF (getprops(cpbar)) THEN
            s(cpbar) = - tint * g(7)
            s(cpbar)    = MIN(s(cpbar),30000.0_sdk)
         END IF
         IF (getprops(hbar)) THEN
            s(hbar) = g(1) - tint * g(4)
         END IF
!
         IF (getprops(qualbar)) THEN
            SELECT CASE (fluidState)
            CASE (vapor)
               s(qualbar) = 1.0_sdk
            CASE (liquid)
               s(qualbar) = 0.0_sdk
!gam  do not return a value for s(qual) when supercritical
            CASE (supercritical)
!gam  set sat properties equal to single phase properties
!gam  and tsat equal to the temperature
               s(tsatbar)   = s(tempbar)
               s(vsubf)  = s(vbar)
               s(vsubg)  = s(vbar)
               s(usubf)  = s(ubar)
               s(usubg)  = s(ubar)
               s(hsubf)  = s(hbar)
               s(hsubg)  = s(hbar)
               s(betasubf)  = s(betabar)
               s(betasubg)  = s(betabar)
               s(kapasubf)  = s(kapabar)
               s(kapasubg)  = s(kapabar)
               s(cpsubf)    = s(cpbar)
               s(cpsubg)    = s(cpbar)
               s(entpysubf) = s(entpybar)
               s(entpysubg) = s(entpybar)
               s(qualbar)   = 0.0_sdk
            END SELECT
         END IF
!
!  Check interpolated values
!
         CALL checkvalue(fluidState, p, g, s, getprops, err, iterate)
!
999      CONTINUE
!
         RETURN
      END SUBROUTINE getsnglatpu
!      
      SUBROUTINE getindex(tables, z, pointer, lower, upper, index, err, lastIndex)
!
!  Find index in table 1 or 2 of the nearest(lesser) temperature
!  or pressure than the input temperature or pressure, z
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
         REAL(sdk) :: z, tables(:)
         INTEGER(sik) :: i, lower, upper, index, pointer, lastIndex
         LOGICAL :: err, found
!
!  Test to make sure that the lower bound is less than or
!  equal to the upper bound
!
         IF (upper < lower) THEN
            err = .TRUE.
!
!  Test to make sure that the input temperature or pressure is
!  within the requested range
!
         ELSE
            found = .FALSE.
            lastIndex = MAX(lastIndex, lower)
!
!           Check to see if z is above or below the lastIndex used for this table.
            IF (z < tables(pointer + lastIndex)) THEN
!
!              z is below the last index used for this table.  Search for the index for which
!              z is larger than the table value.
               DO i = MIN(MAX(lastIndex - 1_sik, lower), upper - 1_sik), lower, -1_sik
                  IF (z >= tables(pointer+i)) THEN
                     found = .TRUE.
                     index = i
                     EXIT
                  END IF
               END DO
            ELSE IF (z > tables(pointer + lastIndex)) THEN
!
!              z is above the last index used for this table.  Search for the index for which
!              z is smaller than the table value.
               DO i = MAX(MIN(lastIndex + 1_sik, upper), lower + 1_sik), upper, 1_sik
                  IF (z < tables(pointer+i)) THEN
                     found = .TRUE.
                     index = i - 1_sik
                     EXIT
                  END IF
               END DO
            ELSE
!
!              z is equal to tables(pointer+lastIndex)
               index = MIN(lastIndex, upper - 1_sik)
               found = .TRUE.
            END IF
!
!           Check to see if the index was found.
            IF (.NOT. found) THEN
!
!              Check to see if z is equal to the last point in the table.
               IF (z == tables(pointer+upper)) THEN
!
!                 z is eual to the last point in the table.
                  index = upper - 1_sik
                  lastIndex = index
                  err = .FALSE.
               ELSE
!
!                 z is out of bounds.
                  err = .TRUE.
               END IF
            ELSE
!
!              The search found a valid index.
               lastIndex = index
               err = .FALSE.
            END IF
         END IF
!
         IF (index < lower .OR. index > upper) THEN
            err = .TRUE.
         END IF
!
         RETURN
      END SUBROUTINE getindex
!
      SUBROUTINE clearprops(getprops)
!
         IMPLICIT NONE
!
         LOGICAL :: getprops(26)
!
         getprops = .FALSE.
!
         RETURN
      END SUBROUTINE clearprops
!       
      SUBROUTINE checkvalue(fluidState, p, g, s, getprops, err, iterate)
!
!  Cognizant engineer:  rwt
!
!         USE TaskData, ONLY: iterate
         USE EosIAPWSData, ONLY: ptrip, tcrit, pcrit, tmin, tmax, liquid, vapor
         USE EosIAPWSData, ONLY: vbar, presbar, tempbar, cpbar, betabar

         IMPLICIT NONE
!
!
         REAL(sdk) :: cvg, g(11), p, s(:), slope, tnaught, ttest
!
         LOGICAL :: getprops(:), err, iterate
         INTEGER(sik), INTENT(IN) :: fluidState
         CHARACTER(LEN=120) :: ErrString
!
         err = .FALSE.
         iterate = .FALSE.
!
         SELECT CASE (fluidState)
         CASE (liquid)
!
!  Liquid must stay above ice line
!
            IF (s(tempbar) .LT. tmin) THEN
               WRITE (ErrString, 800) s(tempbar)
800            FORMAT ('*checkvalue* frozen liquid :', ' temperature = ', 1 p, e13.4)
               !CALL error(2, ErrString)
               iterate = .TRUE.
            END IF
!
!  Spinodal line test for superheated liquid
!  595.0K is the spinodal line temperature
!  at the triple point pressure
!
            tnaught = 595.0_sdk
            slope = (tcrit-tnaught) / (LOG(pcrit)-LOG(ptrip))
            ttest = tnaught + slope * (LOG(p)-LOG(ptrip))
            IF (s(tempbar) .GT. ttest) THEN
               WRITE (ErrString, 801) s(tempbar), ttest
801            FORMAT ('*checkvalue* liquid temperature > spinoidal limit :', ' temperature, spinoidal temper&
              &ature = ', 1 p, 2e13.4)
               !CALL error(2, ErrString)
               iterate = .TRUE.
            END IF
!
         CASE (vapor)
!
!  Vapor cannot exceed maximum table temperature
!
            IF (s(tempbar) .GT. tmax) THEN
               err = .TRUE.
            END IF
!
!  Spinodal line test for subcooled vapor
!
!  Test against 3 criteria for determining a good fit.
!  (1) specific volume > 0.0
!  (2) specific heat at constant volume > 0.0
!  (3) specific volume*kappa > 0.0
!  These are the same criteria used in the generator.
!
            cvg = - s(tempbar) * (g(7)-(g(5)**2/g(3)))
!rex+ 23 Feb 2001
            IF (cvg .LT. 0.0_sdk .AND. g(3) .LT. 0.0_sdk .AND. s(cpbar) .GT. 0.0_sdk) THEN
!          cvg = cp + t*g(5)**2/g(3)
!          use cp to get cv, assume cp/cv=1.1, solve for g5
               g(5) = SQRT(-0.1_sdk*s(cpbar)*g(3)/s(tempbar))
               s(betabar) = g(5) / g(2)
               cvg = - s(tempbar) * (g(7)-(g(5)**2/g(3)))
            END IF
!rex-
            IF (getprops(vbar) .AND. (s(vbar) .LE. 0.0_sdk) .OR. (cvg .LE. 0.0_sdk) .OR. &
                                                             & (-g(3) .LE. 0.0_sdk)) THEN
               IF (s(vbar) .LE. 0.0_sdk) THEN
                  WRITE (ErrString, 802) s(presbar), s(tempbar), s(vbar)
802               FORMAT ('*checkvalue* vapor specific volume < 0 :', 'p, t, v = ', 1 p, 3e13.4)
                  !CALL error(2, ErrString)
                  iterate = .TRUE.
               ELSE IF (cvg .LE. 0.0_sdk) THEN
                  WRITE (ErrString, 803) s(tempbar), cvg
803               FORMAT ('*checkvalue* vapor C_v < 0 :', ' temperature, CsubV = ', 1 p, 2e13.4)
                  !CALL error(2, ErrString)
                  iterate = .TRUE.
               ELSE IF (-g(3) .LE. 0.0_sdk) THEN
                  WRITE (ErrString, 804) s(tempbar), - g(3)
804               FORMAT ('*checkvalue* vapor specific volume*kappa < 0 :', ' temperature, v*kappa = ', 1 p, &
                 & 2e13.4)
                  !CALL error(2, ErrString)
                  iterate = .TRUE.
               END IF
            END IF
         CASE DEFAULT
!
!  Temperature bounds  test
!
            IF ((s(tempbar) .LT. tmin) .OR. (s(tempbar) .GT. tmax)) THEN
               err = .TRUE.
            END IF
!
         END SELECT
!
         RETURN
      END SUBROUTINE checkvalue
!
      SUBROUTINE cubic(a, x, y, err)
!
!  Evaluates the 1D Hermite interpolating polynomial y(x)
!
!  Cognizant engineer:  rwt 5/25/1999
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Local variables
!
         REAL(sdk) :: a(8), delt, psi0, psi1, dydxi, dydxj, psi0x1, psi1x1, psi0x2, psi1x2, xi, xj, x1, x2, &
        & x, yi, yj, yit, yjt, y, z
         LOGICAL :: err
!***********************************************************************
!
!  Statement functions to define Hermite basis functions
!
         psi0(z) = 1.0_sdk + (z**2*(-3.0_sdk+z*(2.0_sdk)))
         psi1(z) = z * (1.0_sdk+z*(-2.0_sdk+z*(1.0_sdk)))
!***********************************************************************
!
!  Executable statements
!
!      call timstart('cubic')
         err = .FALSE.
         xi = a(1)
         xj = a(5)
         delt = (xj-xi)
         x1 = (x-xi) / delt
         x2 = 1.0_sdk - x1
!***********************************************************************
!
!  Definitions
!
         yi = a(2)
         yj = a(6)
         dydxi = a(3)
         dydxj = a(7)
!
         yit = dydxi * delt
         yjt = dydxj * delt
!
         psi0x1 = psi0(x1)
         psi1x1 = psi1(x1)
         psi0x2 = psi0(x2)
         psi1x2 = psi1(x2)
!***********************************************************************
!
! calculate y at x
!
         y = yi * psi0x1 + yj * psi0x2 + yit * psi1x1 - yjt * psi1x2
!***********************************************************************
!      call timstop('cubic')
         RETURN
      END SUBROUTINE cubic
!      
      SUBROUTINE setpointers(fluidState, ofirstpres, olastpres, ptableprop, stableprop, ntableprop, ptablelimits, &
     & ntablelimits, lastIndex)
!
         USE EosIAPWSData, ONLY: liquid, vapor, supercritical, ofirstsatpres, olastsatpres
         USE EosIAPWSData, ONLY: ptable5, stable5, ntable5, ptable8, ntable8
         USE EosIAPWSData, ONLY: ptable6, stable6, ntable6, ptable9, ntable9
         USE EosIAPWSData, ONLY: ptable7, stable7, ntable7, ptable10, ntable10
         USE EosIAPWSData, ONLY: ofirstsupcritpres, olastsupcritpres
         USE EosIAPWSData, ONLY: lastPLiqInd, lastPVapInd, lastPSupInd
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
         INTEGER(sik) :: ofirstpres, olastpres, ntablelimits, ntableprop, ptablelimits, ptableprop, &
        & stableprop, lastIndex
         INTEGER(sik), INTENT(IN) :: fluidState
         LOGICAL :: err
!
         err = .FALSE.
!
         SELECT CASE (fluidState)
         CASE (liquid)
            ofirstpres = ofirstsatpres
            olastpres = olastsatpres
            ptableprop = ptable5
            stableprop = stable5
            ntableprop = ntable5
            ptablelimits = ptable8
            ntablelimits = ntable8
            lastIndex = lastPLiqInd
         CASE (vapor)
            ofirstpres = 1
            olastpres = olastsatpres
            ptableprop = ptable6
            stableprop = stable6
            ntableprop = ntable6
            ptablelimits = ptable9
            ntablelimits = ntable9
            lastIndex = lastPVapInd
         CASE (supercritical)
            ofirstpres = ofirstsupcritpres
            olastpres = olastsupcritpres
            ptableprop = ptable7
            stableprop = stable7
            ntableprop = ntable7
            ptablelimits = ptable10
            ntablelimits = ntable10
            lastIndex = lastPSupInd
         CASE DEFAULT
            err = .TRUE.
         END SELECT
!
         RETURN
     END SUBROUTINE setpointers
!     
      SUBROUTINE setanchor(itleftphi, itleftplo, itrightplo, itrightphi, itleftlimitplo, itrightlimitplo, &
     & itleftlimitphi, itrightlimitphi, p, plo, phi, itleft, itright, threepointleft, err)
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
         REAL(sdk) :: p, plo, phi
         INTEGER(sik) :: itleftlimitplo, itleftplo, itrightplo, itrightlimitplo, itleftlimitphi, itleftphi, &
        & itrightphi, itrightlimitphi, itleft, itright
         LOGICAL :: err, threepointleft
!
!      call timstart('setanchor')
!
         err = .FALSE.
         threepointleft = .FALSE.
!rex+
!         itrightplo = itleftplo + 1 ! not in RELAP5
!         itrightphi = itleftphi + 1 ! not in RELAP5
!rex-
!
         IF (itleftphi .EQ. itleftplo) THEN
!
!   case 1 - we have a rectangle already
!
            itleft = itleftplo
            itright = itleft + 1
         ELSE IF (itleftphi .GT. itleftplo) THEN
!
!   rhombus tilts to the right
!
            IF (itleftplo .LT. itleftlimitphi) THEN
!
!   bottom left point is under one or more bad points
!
!   case 2a
!
               IF (((p-plo)/(phi-plo)) .LE. 0.5_sdk) THEN
                  threepointleft = .TRUE.
                  itleft = itleftplo
                  itright = itleftlimitphi
                  IF (itright .GT. itrightlimitplo) err = .TRUE.
               ELSE
                  CALL moveloright(itleftplo, itleftphi)
                  itleft = itleftplo
                  itright = itleft + 1
                  IF (itright .GT. itrightlimitplo) err = .TRUE.
               END IF
!
            ELSE IF (itrightplo .EQ. itrightlimitplo) THEN
!
!   case 3 - limit at bottom right
!
               CALL movehileft(itleftphi, itleftplo)
               itleft = itleftplo
               itright = itleft + 1
               IF (itleft .LT. itleftlimitphi) err = .TRUE.
!
            ELSE IF (((p-plo)/(phi-plo)) .LE. 0.5_sdk) THEN
!
!   case 4A - pin is closer to plo
!
               CALL movehileft(itleftphi, itleftplo)
               itleft = itleftplo
               itright = itleft + 1
!
            ELSE
!
!   case 4B - pin is closer to phi
!
               CALL moveloright(itleftplo, itleftphi)
               itleft = itleftplo
               itright = itleft + 1
!   check for cases when lower right point of rhombus is
!   shifted left by more than two points than the upper right point
               IF (itright .GT. itrightlimitplo) THEN
                  itright = itrightlimitplo
                  itleft = itright - 1
               END IF
               IF (itright .GT. itrightlimitplo) err = .TRUE.
            END IF
!
         ELSE IF (itleftphi .LT. itleftplo) THEN
!
!   rhombus tilts to the left
!
            IF (itrightphi .EQ. itrightlimitphi) THEN
!
!   case 5 - limit at top left
!
               CALL moveloleft(itleftplo, itleftphi)
               itleft = itleftplo
               itright = itleft + 1
               IF (itleft .LT. itleftlimitplo) err = .TRUE.
            ELSE IF (itleftplo .EQ. itleftlimitplo) THEN
!
!   case 6 - limit at bottom left
!
               CALL movehiright(itleftphi, itleftplo)
               itleft = itleftphi
               itright = itleft + 1
               IF (itright .GT. itrightlimitphi) err = .TRUE.
!
            ELSE IF (((p-plo)/(phi-plo)) .LE. 0.5_sdk) THEN
!
!   case 7A - pin is ploser to plo
!
               CALL movehiright(itleftphi, itleftplo)
               itleft = itleftphi
               itright = itleft + 1
!   check for cases when upper right point of rhombus is
!   shifted left by more than two points than the lower right point
               IF (itright .GT. itrightlimitphi) THEN
                  itright = itrightlimitphi
                  itleft = itright - 1
!            extrapflaglo = 'right'
               END IF
               IF (itright .GT. itrightlimitphi) err = .TRUE.
!
            ELSE
!
!   case 7B - pin is ploser to plo
!
               CALL moveloleft(itleftplo, itleftphi)
               itleft = itleftplo
               itright = itleft + 1
!   check for cases when lower left point of rhombus is
!   shifted right by more than two points than the upper left point
               IF (itleft .LT. itleftlimitplo) THEN
                  itleft = itleftlimitplo
                  itright = itleft + 1
!            extrapflaglo = 'left'
               END IF
               IF (itleft .LT. itleftlimitplo) err = .TRUE.
            END IF
         END IF
!
         RETURN
     END SUBROUTINE setanchor
!
      SUBROUTINE loadcorners(tables, a, itleft, itright, iplo, ptableprop, stableprop, ntableprop, &
     & ofirstpres, ptr)
!
!  Loads the information needed at the four corner points for
!  the four-point(2D) Hermite interpolation formula for
!  liquid
!
!  Cognizant engineer:  rwt
!
         USE EosIAPWSData, ONLY: ptable1, ptable2
!
         IMPLICIT NONE
!
         REAL(sdk) :: a(11, 4), tables(:)
!      real f(36)
         INTEGER(sik) :: p1, p2, t1, t2
         INTEGER(sik) :: c(4), iplo, itleft, ntableprop, ofirstpres, ptableprop, stableprop, ptr, itright
!
!      call timstart('loadcorners')
!
!  Load pressure and temperature indices
!
         t1 = ptable1 + itleft
         t2 = ptable1 + itright
         p1 = ptable2 + iplo
         p2 = ptable2 + iplo + 1
!
!   Load corner point info for the 4-point formula
!   c(1) is lower left hand corner
!   c(2) is lower right hand corner
!   c(3) is upper left hand corner
!   c(4) is upper right hand corner
!
         c(1) = ptableprop + (iplo-ofirstpres) * stableprop + (itleft-1) * ntableprop
!      ptr = c(1)+11
         ptr = c(1) + 1
!
!  Load up the a array for the 4-point formula
!
         a(1, 1) = tables(t1)
         a(1, 2) = tables(t2)
         a(1, 3) = tables(t1)
         a(1, 4) = tables(t2)
         a(2, 1) = tables(p1)
         a(2, 2) = tables(p1)
         a(2, 3) = tables(p2)
         a(2, 4) = tables(p2)
!
!
!      call timstop('loadcorners')
!
         RETURN
     END SUBROUTINE loadcorners
!
!##########################################################
      SUBROUTINE movehileft(itleftphi, itleftplo)
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
!
!
!
!
         INTEGER(sik) :: itleftplo, itleftphi
!
         itleftphi = itleftplo
!
         RETURN
      END SUBROUTINE movehileft
!##############################################################
      SUBROUTINE movehiright(itleftphi, itleftplo)
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
!
!
!
!
         INTEGER(sik) :: itleftplo, itleftphi
!
         itleftphi = itleftplo
!
         RETURN
      END SUBROUTINE movehiright
!#############################################################     
      SUBROUTINE moveloleft(itleftplo, itleftphi)
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
!
!
!
!
         INTEGER(sik) :: itleftplo, itleftphi
!
         itleftplo = itleftphi
!
         RETURN
      END SUBROUTINE moveloleft
!################################################################
      SUBROUTINE moveloright(itleftplo, itleftphi)
!
!  Cognizant engineer:  rwt
!
         IMPLICIT NONE
!
!
!
!
!
         INTEGER(sik) :: itleftplo, itleftphi
!
         itleftplo = itleftphi
!
         RETURN
      END SUBROUTINE moveloright  
!###########################################################
      SUBROUTINE Beta2d(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rex 3/28/2001
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj, tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, d2gtildedpdt
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!***********************************************************************
!
!  Statement functions to define Hermite polynomials
!
!***********************************************************************
!
!  Definitions
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         pii = a(2, 1)
!         pij = a(2, 2)
         pji = a(2, 3)
!         pjj = a(2, 4)
!
         deltt = (tij-tii)
         x = (t-tii) / deltt
         IF (fluidState == vapor) THEN
            deltp = (LOG(pji)-LOG(pii))
            y = (LOG(p)-LOG(pii)) / deltp
         ELSE
            deltp = (pji-pii)
            y = (p-pii) / deltp
         END IF
!
!
!***********************************************************************
! calculate gtilde
!
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
!
!***********************************************************************
! calculate d2(gtilde)/dtdp
         d2gtildedpdt = gptfunct(yy, xx, tables, ptr)
         d2gtildedpdt = d2gtildedpdt / (deltp*deltt)
!***********************************************************************
!
         IF (fluidState == vapor) THEN
            d2gtildedpdt = d2gtildedpdt / p
         END IF
!  fill in g array
         g(5) = d2gtildedpdt
!
         RETURN
      END SUBROUTINE Beta2d
!###########################################################
      SUBROUTINE herm2d(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rwt 5/25/1999
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj, tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, gtilde, dgtildedt, dgtildedp, d2gtildedt2, d2gtildedp2, d2gtildedpdt
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!***********************************************************************
!
!  Statement functions to define Hermite polynomials
!
!***********************************************************************
!
!  Definitions
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         pii = a(2, 1)
!         pij = a(2, 2)
         pji = a(2, 3)
!         pjj = a(2, 4)
!
         deltt = (tij-tii)
         x = (t-tii) / deltt
         IF (fluidState == vapor) THEN
            deltp = (LOG(pji)-LOG(pii))
            y = (LOG(p)-LOG(pii)) / deltp
         ELSE
            deltp = (pji-pii)
            y = (p-pii) / deltp
         END IF
!
!
!***********************************************************************
! calculate gtilde
!
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
         gtilde = gfunct(yy, xx, tables, ptr)
!***********************************************************************
! calculate d(gtilde)/dt
!
         dgtildedt = gtfunct(yy, xx, tables, ptr)
         dgtildedt = dgtildedt / deltt
!***********************************************************************
! calculate d(gtilde)/dp
!
         dgtildedp = gpfunct(yy, xx, tables, ptr)
         dgtildedp = dgtildedp / deltp
!***********************************************************************
! calculate d2(gtilde)/dt2
         d2gtildedt2 = gttfunct(yy, xx, tables, ptr)
         d2gtildedt2 = d2gtildedt2 / deltt ** 2
!***********************************************************************
! calculate d2(gtilde)/dp2
!
         d2gtildedp2 = gppfunct(yy, xx, tables, ptr)
         d2gtildedp2 = d2gtildedp2 / deltp ** 2
!***********************************************************************
! calculate d2(gtilde)/dtdp
         d2gtildedpdt = gptfunct(yy, xx, tables, ptr)
         d2gtildedpdt = d2gtildedpdt / (deltp*deltt)
!***********************************************************************
!
         IF (fluidState == vapor) THEN
            d2gtildedp2 = (d2gtildedp2-dgtildedp) / p ** 2
            dgtildedp = dgtildedp / p
            d2gtildedpdt = d2gtildedpdt / p
         END IF
!  fill in g array
         g(1) = gtilde
         g(2) = dgtildedp
         g(3) = d2gtildedp2
         g(4) = dgtildedt
         g(5) = d2gtildedpdt
         g(7) = d2gtildedt2
!
         RETURN
      END SUBROUTINE herm2d
!###########################################################
      SUBROUTINE herm2dleft(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rwt 5/25/1999
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj, tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, gtilde, dgtildedt, dgtildedp, d2gtildedt2, d2gtildedp2, d2gtildedpdt
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!***********************************************************************
!
!  Statement functions to define Hermite polynomials
!
!***********************************************************************
!
!  Definitions
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         IF (fluidState == vapor) THEN
            p   = LOG(p)
            pii = LOG(a(2, 1))
!            pij = LOG(a(2, 2))
            pji = LOG(a(2, 3))
!            pjj = LOG(a(2, 4))
         ELSE
            pii = a(2, 1)
!            pij = a(2, 2)
            pji = a(2, 3)
!            pjj = a(2, 4)
         END IF
!
       deltt = (tij - tii)
       deltp = (pji - pii)
       x     = (t - tii)/deltt
       y     = (p - pii)/deltp
!
!***********************************************************************
! calculate gtilde
!
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
         gtilde = gfuncte(yy, xx, tables, ptr)
!***********************************************************************
! calculate d(gtilde)/dt
!
         dgtildedt = gtfuncte(yy, xx, tables, ptr)
         dgtildedt = dgtildedt / deltt
!***********************************************************************
! calculate d(gtilde)/dp
!
         dgtildedp = gpfuncte(yy, xx, tables, ptr)
         dgtildedp = dgtildedp / deltp
!***********************************************************************
! calculate d2(gtilde)/dt2
         d2gtildedt2 = gttfuncte(yy, xx, tables, ptr)
         d2gtildedt2 = d2gtildedt2 / deltt ** 2
!***********************************************************************
! calculate d2(gtilde)/dp2
!
         d2gtildedp2 = gppfuncte(yy, xx, tables, ptr)
         d2gtildedp2 = d2gtildedp2 / deltp ** 2
!***********************************************************************
! calculate d2(gtilde)/dtdp
         d2gtildedpdt = gptfuncte(yy, xx, tables, ptr)
         d2gtildedpdt = d2gtildedpdt / (deltp*deltt)
!***********************************************************************
         IF (fluidState == vapor) THEN
            p = EXP(p)
            d2gtildedp2 = (d2gtildedp2-dgtildedp) / p ** 2
            dgtildedp = dgtildedp / p
            d2gtildedpdt = d2gtildedpdt / p
         END IF
!  fill in g array
         g(1) = gtilde
         g(2) = dgtildedp
         g(3) = d2gtildedp2
         g(4) = dgtildedt
         g(5) = d2gtildedpdt
         g(7) = d2gtildedt2
!
         RETURN
      END SUBROUTINE herm2dleft
!##############################################################
      SUBROUTINE herm2dplus(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rwt 5/25/1999
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
         REAL(sdk) :: a, deltt, deltp, gtilde, dgtildedt, dgtildedp, d2gtildedt2, d2gtildedpdt, g, p, pii, &
        & pji, tii, tij, t
!        & pij, pji, pjj, tii, tij, tji, tjj, t
         DIMENSION a(11, 4), g(11)
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!***********************************************************************
!
!  Statement functions to define Hermite polynomials
!
!***********************************************************************
!
!  Definitions
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
!
         deltt = (tij - tii)
         x = (t - tii) / deltt
         IF (fluidState == vapor) THEN
            p   = LOG(p)
            pii = LOG(a(2, 1))
!            pij = LOG(a(2, 2))
            pji = LOG(a(2, 3))
!            pjj = LOG(a(2, 4))
         ELSE
            pii = a(2, 1)
!            pij = a(2, 2)
            pji = a(2, 3)
!            pjj = a(2, 4)
         END IF
         deltt = (tij - tii)
         deltp = (pji - pii)
         x = (t - tii) / deltt
         y = (p-pii) / deltp
!
!***********************************************************************
! calculate gtilde
!
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
         gtilde = gfunct(yy, xx, tables, ptr)
!***********************************************************************
! calculate d(gtilde)/dt
!
         dgtildedt = gtfunct(yy, xx, tables, ptr)
         dgtildedt = dgtildedt / deltt
! calculate d(gtilde)/dp
         dgtildedp = gpfunct(yy, xx, tables, ptr)
         dgtildedp = dgtildedp / deltp
         d2gtildedt2 = gttfunct(yy, xx, tables, ptr)
         d2gtildedt2 = d2gtildedt2 / deltt ** 2
         d2gtildedpdt = gptfunct(yy, xx, tables, ptr)
         d2gtildedpdt = d2gtildedpdt / (deltp*deltt)
         IF (fluidState == vapor) THEN
            p = EXP(p)
            dgtildedp = dgtildedp / p
            d2gtildedpdt = d2gtildedpdt / p
         END IF
!  fill in g array
         g(1) = gtilde
         g(2) = dgtildedp
         g(4) = dgtildedt
         g(5) = d2gtildedpdt
         g(7) = d2gtildedt2
!
         RETURN
      END SUBROUTINE herm2dplus
!##############################################################      
      SUBROUTINE Betaline(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rex 3/20/2001
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj, tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, d2gtildedpdt
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         pii = a(2, 1)
!         pij = a(2, 2)
         pji = a(2, 3)
!         pjj = a(2, 4)
!
         deltt = (tij-tii)
         x = (t-tii) / deltt
         IF (fluidState == vapor) THEN
            deltp = (LOG(pji)-LOG(pii))
            y = (LOG(p)-LOG(pii)) / deltp
         ELSE
            deltp = (pji-pii)
            y = (p-pii) / deltp
         END IF
!
! calculate gtilde
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
!***********************************************************************
! calculate d2(gtilde)/dtdp
         d2gtildedpdt = gptfunct(yy, xx, tables, ptr)
         d2gtildedpdt = d2gtildedpdt / (deltp*deltt)
!***********************************************************************
!
         IF (fluidState == vapor) THEN
            d2gtildedpdt = d2gtildedpdt / p
         END IF
!
         g(5) = d2gtildedpdt
!
         RETURN
      END SUBROUTINE Betaline
!###########################################################
      SUBROUTINE Cpline(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rex 3/28/2001
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj,  tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, d2gtildedt2
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         pii = a(2, 1)
!         pij = a(2, 2)
         pji = a(2, 3)
!         pjj = a(2, 4)
!
         deltt = (tij-tii)
         deltp = (pji-pii)
         x = (t-tii) / deltt
         y = (p-pii) / deltp
         IF (fluidState == vapor) THEN
            deltp = (LOG(pji)-LOG(pii))
            y = (LOG(p)-LOG(pii)) / deltp
         END IF
!
! calculate gtilde
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
!***********************************************************************
! calculate d2(gtilde)/dtdp
         d2gtildedt2 = gttfunctline(yy, xx, tables, ptr)
         d2gtildedt2 = d2gtildedt2 / (deltt*deltt)
!***********************************************************************
!
         g(7) = d2gtildedt2
!
         RETURN
      END SUBROUTINE Cpline
!###########################################################
      SUBROUTINE Kappaline(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rex 3/20/2001
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj, tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, d2gtildedp2, dgtildedp
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         pii = a(2, 1)
!         pij = a(2, 2)
         pji = a(2, 3)
!         pjj = a(2, 4)
!
         deltt = (tij-tii)
         deltp = (pji-pii)
         x = (t-tii) / deltt
         y = (p-pii) / deltp
         IF (fluidState == vapor) THEN
            deltp = (LOG(pji)-LOG(pii))
            y = (LOG(p)-LOG(pii)) / deltp
         END IF
!
! calculate gtilde
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
!
! calculate d(gtilde)/dp
!
         dgtildedp = gpfunctline(yy, xx, tables, ptr)
         dgtildedp = dgtildedp / deltp
!
! calculate d2(gtilde)/dp2
!
         d2gtildedp2 = gppfunctline(yy, xx, tables, ptr)
         d2gtildedp2 = d2gtildedp2 / deltp ** 2
!
!
         IF (fluidState == vapor) THEN
            d2gtildedp2 = (d2gtildedp2-dgtildedp) / p ** 2
         END IF
!
         g(3) = d2gtildedp2
!
         RETURN
      END SUBROUTINE Kappaline
!###########################################################
      SUBROUTINE Rholine(tables, ptr, a, fluidState, g, t, p, err)
!
         USE EosIAPWSData, ONLY: vapor
!
!  Evaluates the 2D Hermite interpolating polynomial and several
!  derivatives
!
!  Cognizant engineer:  rex 3/28/2001
!
         IMPLICIT NONE
!
!***********************************************************************
!
!  Declarations
!
         INTEGER(sik) :: ptr
         REAL(sdk) :: xx(6), yy(6), tables(:)
         REAL(sdk) :: x, y
!         REAL(sdk) :: p, t, pii, pij, pji, pjj, tii, tij, tji, tjj
         REAL(sdk) :: p, t, pii, pji, tii, tij
         REAL(sdk) :: deltt, deltp, dgtildedp
         REAL(sdk), DIMENSION(11, 4) :: a
         REAL(sdk), DIMENSION(11) :: g
         LOGICAL :: err
         INTEGER(sik), INTENT(IN) :: fluidState
!
         err = .FALSE.
!
         tii = a(1, 1)
         tij = a(1, 2)
!         tji = a(1, 3)
!         tjj = a(1, 4)
         pii = a(2, 1)
!         pij = a(2, 2)
         pji = a(2, 3)
!         pjj = a(2, 4)
!
         deltt = (tij-tii)
         deltp = (pji-pii)
         x = (t-tii) / deltt
         y = (p-pii) / deltp
         IF (fluidState == vapor) THEN
            deltp = (LOG(pji)-LOG(pii))
            y = (LOG(p)-LOG(pii)) / deltp
         END IF
!
! calculate gtilde
         xx(1) = 1.0_sdk
         xx(2) = x
         xx(3) = x ** 2
         xx(4) = x ** 3
         xx(5) = x ** 4
         xx(6) = x ** 5
         yy(1) = 1.0_sdk
         yy(2) = y
         yy(3) = y ** 2
         yy(4) = y ** 3
         yy(5) = y ** 4
         yy(6) = y ** 5
!
! calculate d(gtilde)/dp
!
         dgtildedp = gpfunctline(yy, xx, tables, ptr)
         dgtildedp = dgtildedp / deltp
!
         IF (fluidState .EQ. vapor) THEN
            dgtildedp = dgtildedp / p
         END IF
!
         g(2) = dgtildedp
!
         RETURN
      END SUBROUTINE Rholine
!##############################################
!##############################################
      REAL(sdkx) FUNCTION gfunct(deltapres, deltatemp, tables, ptr)
!
!  Evaluate g as a function of dimensionless deltapres and deltatemp giv
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltatemp
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         INTEGER(sik) :: ptr
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
!
             gfunct = deltapres(1)*(tables(ptr+ 1)*deltatemp(1)     &
           &        +               tables(ptr+ 2)*deltatemp(2)     &
           &        +               tables(ptr+ 3)*deltatemp(3)     &
           &        +               tables(ptr+ 4)*deltatemp(4)     &
           &        +               tables(ptr+ 5)*deltatemp(5)     &
           &        +               tables(ptr+ 6)*deltatemp(6))    &
           &        + deltapres(2)*(tables(ptr+ 7)*deltatemp(1)     &
           &        +               tables(ptr+ 8)*deltatemp(2)     &
           &        +               tables(ptr+ 9)*deltatemp(3)     &
           &        +               tables(ptr+10)*deltatemp(4)     &
           &        +               tables(ptr+11)*deltatemp(5)     &
           &        +               tables(ptr+12)*deltatemp(6))    &
           &        + deltapres(3)*(tables(ptr+13)*deltatemp(1)     &
           &        +               tables(ptr+14)*deltatemp(2)     &
           &        +               tables(ptr+15)*deltatemp(3)     &
           &        +               tables(ptr+16)*deltatemp(4)     &
           &        +               tables(ptr+17)*deltatemp(5)     &
           &        +               tables(ptr+18)*deltatemp(6))
             gfunct = gfunct                                        &
           &        + deltapres(4)*(tables(ptr+19)*deltatemp(1)     &
           &        +               tables(ptr+20)*deltatemp(2)     &
           &        +               tables(ptr+21)*deltatemp(3)     &
           &        +               tables(ptr+22)*deltatemp(4)     &
           &        +               tables(ptr+23)*deltatemp(5)     &
           &        +               tables(ptr+24)*deltatemp(6))    &
           &        + deltapres(5)*(tables(ptr+25)*deltatemp(1)     &
           &        +               tables(ptr+26)*deltatemp(2)     &
           &        +               tables(ptr+27)*deltatemp(3)     &
           &        +               tables(ptr+28)*deltatemp(4)     &
           &        +               tables(ptr+29)*deltatemp(5)     &
           &        +               tables(ptr+30)*deltatemp(6))    &
           &        + deltapres(6)*(tables(ptr+31)*deltatemp(1)     &
           &        +               tables(ptr+32)*deltatemp(2)     &
           &        +               tables(ptr+33)*deltatemp(3)     &
           &        +               tables(ptr+34)*deltatemp(4)     &
           &        +               tables(ptr+35)*deltatemp(5)     &
           &        +               tables(ptr+36)*deltatemp(6))
!
         RETURN
      END FUNCTION gfunct
!##############################################
      REAL(sdkx) FUNCTION gfuncte(deltapres, deltatemp, tables, ptr)
!
!  Evaluate g as a function of dimensionless deltapres and deltatemp giv
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltatemp
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         INTEGER(sik) :: ptr
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
!
             gfuncte = deltapres(1)*(tables(ptr+ 1)*deltatemp(1)     &
           &         +               tables(ptr+ 2)*deltatemp(2)     &
           &         +               tables(ptr+ 3)*deltatemp(3)     &
           &         +               tables(ptr+ 4)*deltatemp(4))    &
           &         + deltapres(2)*(tables(ptr+ 7)*deltatemp(1)     &
           &         +               tables(ptr+ 8)*deltatemp(2)     &
           &         +               tables(ptr+ 9)*deltatemp(3)     &
           &         +               tables(ptr+10)*deltatemp(4))    &
           &         + deltapres(3)*(tables(ptr+13)*deltatemp(1)     &
           &         +               tables(ptr+14)*deltatemp(2)     &
           &         +               tables(ptr+15)*deltatemp(3)     &
           &         +               tables(ptr+16)*deltatemp(4))
             gfuncte = gfuncte                                       &
           &         + deltapres(4)*(tables(ptr+19)*deltatemp(1)     &
           &         +               tables(ptr+20)*deltatemp(2)     &
           &         +               tables(ptr+21)*deltatemp(3)     &
           &         +               tables(ptr+22)*deltatemp(4))    &
           &         + deltapres(5)*(tables(ptr+25)*deltatemp(1)     &
           &         +               tables(ptr+26)*deltatemp(2)     &
           &         +               tables(ptr+27)*deltatemp(3)     &
           &         +               tables(ptr+28)*deltatemp(4))    &
           &         + deltapres(6)*(tables(ptr+31)*deltatemp(1)     &
           &         +               tables(ptr+32)*deltatemp(2)     &
           &         +               tables(ptr+33)*deltatemp(3)     &
           &         +               tables(ptr+34)*deltatemp(4))
!
         RETURN
      END FUNCTION gfuncte
!##############################################
      REAL(sdkx) FUNCTION gfunctline(deltapres, deltatemp, tables, ptr)
!
!  Evaluate g as a function of dimensionless deltapres and deltatemp giv
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltatemp
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         INTEGER(sik) :: ptr
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
!
             gfunctline = deltapres(1)*(tables(ptr+ 1)*deltatemp(1)      &
           &            +               tables(ptr+ 2)*deltatemp(2)      &
           &            +               tables(ptr+ 3)*deltatemp(3))     &
           &            + deltapres(2)*(tables(ptr+ 7)*deltatemp(1)      &
           &            +               tables(ptr+ 8)*deltatemp(2)      &
           &            +               tables(ptr+ 9)*deltatemp(3))     &
           &            + deltapres(3)*(tables(ptr+13)*deltatemp(1)      &
           &            +               tables(ptr+14)*deltatemp(2)      &
           &            +               tables(ptr+15)*deltatemp(3))
         RETURN
      END FUNCTION gfunctline
!##################################################
      REAL(sdkx) FUNCTION gpfunct(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gp(dg/dp) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
!
         INTEGER(sik) :: ptr
!
!
             gpfunct = 1.0d0*deltapres(1)*(tables(ptr+ 7)*deltatemp(1)     &
           &         +                   tables(ptr+ 8)*deltatemp(2)       &
           &         +                   tables(ptr+ 9)*deltatemp(3)       &
           &         +                   tables(ptr+10)*deltatemp(4)       &
           &         +                   tables(ptr+11)*deltatemp(5)       &
           &         +                   tables(ptr+12)*deltatemp(6))      &
           &         + 2.0d0*deltapres(2)*(tables(ptr+13)*deltatemp(1)     &
           &         +                   tables(ptr+14)*deltatemp(2)       &
           &         +                   tables(ptr+15)*deltatemp(3)       &
           &         +                   tables(ptr+16)*deltatemp(4)       &
           &         +                   tables(ptr+17)*deltatemp(5)       &
           &         +                   tables(ptr+18)*deltatemp(6))      &
           &         + 3.0d0*deltapres(3)*(tables(ptr+19)*deltatemp(1)     &
           &         +                   tables(ptr+20)*deltatemp(2)       &
           &         +                   tables(ptr+21)*deltatemp(3)       &
           &         +                   tables(ptr+22)*deltatemp(4)       &
           &         +                   tables(ptr+23)*deltatemp(5)       &
           &         +                   tables(ptr+24)*deltatemp(6))
             gpfunct = gpfunct                                             &
           &         + 4.0d0*deltapres(4)*(tables(ptr+25)*deltatemp(1)     &
           &         +                   tables(ptr+26)*deltatemp(2)       &
           &         +                   tables(ptr+27)*deltatemp(3)       &
           &         +                   tables(ptr+28)*deltatemp(4)       &
           &         +                   tables(ptr+29)*deltatemp(5)       &
           &         +                   tables(ptr+30)*deltatemp(6))      &
           &         + 5.0d0*deltapres(5)*(tables(ptr+31)*deltatemp(1)     &
           &         +                   tables(ptr+32)*deltatemp(2)       &
           &         +                   tables(ptr+33)*deltatemp(3)       &
           &         +                   tables(ptr+34)*deltatemp(4)       &
           &         +                   tables(ptr+35)*deltatemp(5)       &
           &         +                   tables(ptr+36)*deltatemp(6))
         RETURN
      END FUNCTION gpfunct
!##################################################
      REAL(sdkx) FUNCTION gpfuncte(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gp(dg/dp) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
!
         INTEGER(sik) :: ptr
!
         gpfuncte = 1.0d0*deltapres(1)*(tables(ptr+ 7)*deltatemp(1)  &
       &          +                     tables(ptr+ 8)*deltatemp(2)  &
       &          +                     tables(ptr+ 9)*deltatemp(3)  &
       &          +                     tables(ptr+10)*deltatemp(4)) &
       &          + 2.0d0*deltapres(2)*(tables(ptr+13)*deltatemp(1)  &
       &          +                     tables(ptr+14)*deltatemp(2)  &
       &          +                     tables(ptr+15)*deltatemp(3)  &
       &          +                     tables(ptr+16)*deltatemp(4)) &
       &          + 3.0d0*deltapres(3)*(tables(ptr+19)*deltatemp(1)  &
       &          +                     tables(ptr+20)*deltatemp(2)  &
       &          +                     tables(ptr+21)*deltatemp(3)  &
       &          +                     tables(ptr+22)*deltatemp(4))
         gpfuncte = gpfuncte                                         &
       &          + 4.0d0*deltapres(4)*(tables(ptr+25)*deltatemp(1)  &
       &          +                     tables(ptr+26)*deltatemp(2)  &
       &          +                     tables(ptr+27)*deltatemp(3)  &
       &          +                     tables(ptr+28)*deltatemp(4)) &
       &          + 5.0d0*deltapres(5)*(tables(ptr+31)*deltatemp(1)  &
       &          +                     tables(ptr+32)*deltatemp(2)  &
       &          +                     tables(ptr+33)*deltatemp(3)  &
       &          +                     tables(ptr+34)*deltatemp(4))

         RETURN
      END FUNCTION gpfuncte
!##################################################
      REAL(sdkx) FUNCTION gpfunctline(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gp(dg/dp) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
!
         INTEGER(sik) :: ptr
!
!
             gpfunctline = 1.0d0*deltapres(1)*(tables(ptr+ 7)*deltatemp(1)     &
           &             +                   tables(ptr+ 8)*deltatemp(2)       &
           &             +                   tables(ptr+ 9)*deltatemp(3))      &
           &             + 2.0d0*deltapres(2)*(tables(ptr+13)*deltatemp(1)     &
           &             +                   tables(ptr+14)*deltatemp(2)       &
           &             +                   tables(ptr+15)*deltatemp(3))
!
         RETURN
      END FUNCTION gpfunctline
!#######################################
      REAL(sdkx) FUNCTION gppfunct(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gpp(d2g/dp2) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
             gppfunct = 2.0d0*1.0d0*deltapres(1)*(tables(ptr+13)*deltatemp(1)     &
           &          +                       tables(ptr+14)*deltatemp(2)         &
           &          +                       tables(ptr+15)*deltatemp(3)         &
           &          +                       tables(ptr+16)*deltatemp(4)         &
           &          +                       tables(ptr+17)*deltatemp(5)         &
           &          +                       tables(ptr+18)*deltatemp(6))        &
           &          + 3.0d0*2.0d0*deltapres(2)*(tables(ptr+19)*deltatemp(1)     &
           &          +                       tables(ptr+20)*deltatemp(2)         &
           &          +                       tables(ptr+21)*deltatemp(3)         &
           &          +                       tables(ptr+22)*deltatemp(4)         &
           &          +                       tables(ptr+23)*deltatemp(5)         &
           &          +                       tables(ptr+24)*deltatemp(6))
             gppfunct = gppfunct                                                  &
           &          + 4.0d0*3.0d0*deltapres(3)*(tables(ptr+25)*deltatemp(1)     &
           &          +                       tables(ptr+26)*deltatemp(2)         &
           &          +                       tables(ptr+27)*deltatemp(3)         &
           &          +                       tables(ptr+28)*deltatemp(4)         &
           &          +                       tables(ptr+29)*deltatemp(5)         &
           &          +                       tables(ptr+30)*deltatemp(6))        &
           &          + 5.0d0*4.0d0*deltapres(4)*(tables(ptr+31)*deltatemp(1)     &
           &          +                       tables(ptr+32)*deltatemp(2)         &
           &          +                       tables(ptr+33)*deltatemp(3)         &
           &          +                       tables(ptr+34)*deltatemp(4)         &
           &          +                       tables(ptr+35)*deltatemp(5)         &
           &          +                       tables(ptr+36)*deltatemp(6))
!
         RETURN
      END FUNCTION gppfunct
!#################################################################
      REAL(sdkx) FUNCTION gppfuncte(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gpp(d2g/dp2) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
             gppfuncte = 2.0d0*1.0d0*deltapres(1)*(tables(ptr+13)*deltatemp(1)     &
           &           +                       tables(ptr+14)*deltatemp(2)         &
           &           +                       tables(ptr+15)*deltatemp(3)         &
           &           +                       tables(ptr+16)*deltatemp(4))        &
           &           + 3.0d0*2.0d0*deltapres(2)*(tables(ptr+19)*deltatemp(1)     &
           &           +                       tables(ptr+20)*deltatemp(2)         &
           &           +                       tables(ptr+21)*deltatemp(3)         &
           &           +                       tables(ptr+22)*deltatemp(4))
             gppfuncte = gppfuncte                                                 &
           &           + 4.0d0*3.0d0*deltapres(3)*(tables(ptr+25)*deltatemp(1)     &
           &           +                       tables(ptr+26)*deltatemp(2)         &
           &           +                       tables(ptr+27)*deltatemp(3)         &
           &           +                       tables(ptr+28)*deltatemp(4))        &
           &           + 5.0d0*4.0d0*deltapres(4)*(tables(ptr+31)*deltatemp(1)     &
           &           +                       tables(ptr+32)*deltatemp(2)         &
           &           +                       tables(ptr+33)*deltatemp(3)         &
           &           +                       tables(ptr+34)*deltatemp(4))
!
         RETURN
      END FUNCTION gppfuncte
!#################################################################
      REAL(sdkx) FUNCTION gppfunctline(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gpp(d2g/dp2) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
!
!
!
             gppfunctline =                                               &
           &       2.0d0*1.0d0*deltapres(1)*(tables(ptr+13)*deltatemp(1)  &
           &                           + tables(ptr+14)*deltatemp(2)      &
           &                           + tables(ptr+15)*deltatemp(3))
         RETURN
      END FUNCTION gppfunctline
!#################################################
      REAL(sdkx) FUNCTION gptfunct(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gpt(d2g/dpdt) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
             gptfunct = 1.0d0*deltapres(1)*(tables(ptr+ 8)*1.0d0*deltatemp(1)     &
           &          +                   tables(ptr+ 9)*2.0d0*deltatemp(2)       &
           &          +                   tables(ptr+10)*3.0d0*deltatemp(3)       &
           &          +                   tables(ptr+11)*4.0d0*deltatemp(4)       &
           &          +                   tables(ptr+12)*5.0d0*deltatemp(5))      &
           &          + 2.0d0*deltapres(2)*(tables(ptr+14)*1.0d0*deltatemp(1)     &
           &          +                   tables(ptr+15)*2.0d0*deltatemp(2)       &
           &          +                   tables(ptr+16)*3.0d0*deltatemp(3)       &
           &          +                   tables(ptr+17)*4.0d0*deltatemp(4)       &
           &          +                   tables(ptr+18)*5.0d0*deltatemp(5))      &
           &          + 3.0d0*deltapres(3)*(tables(ptr+20)*1.0d0*deltatemp(1)     &
           &          +                   tables(ptr+21)*2.0d0*deltatemp(2)       &
           &          +                   tables(ptr+22)*3.0d0*deltatemp(3)       &
           &          +                   tables(ptr+23)*4.0d0*deltatemp(4)       &
           &          +                   tables(ptr+24)*5.0d0*deltatemp(5))
             gptfunct = gptfunct                                                  &
           &          + 4.0d0*deltapres(4)*(tables(ptr+26)*1.0d0*deltatemp(1)     &
           &          +                   tables(ptr+27)*2.0d0*deltatemp(2)       &
           &          +                   tables(ptr+28)*3.0d0*deltatemp(3)       &
           &          +                   tables(ptr+29)*4.0d0*deltatemp(4)       &
           &          +                   tables(ptr+30)*5.0d0*deltatemp(5))      &
           &          + 5.0d0*deltapres(5)*(tables(ptr+32)*1.0d0*deltatemp(1)     &
           &          +                   tables(ptr+33)*2.0d0*deltatemp(2)       &
           &          +                   tables(ptr+34)*3.0d0*deltatemp(3)       &
           &          +                   tables(ptr+35)*4.0d0*deltatemp(4)       &
           &          +                   tables(ptr+36)*5.0d0*deltatemp(5))
!
         RETURN
      END FUNCTION gptfunct
!#################################################
      REAL(sdkx) FUNCTION gptfuncte(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gpt(d2g/dpdt) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
             gptfuncte = 1.0d0*deltapres(1)*(tables(ptr+ 8)*1.0d0*deltatemp(1)    &
           &           +                   tables(ptr+ 9)*2.0d0*deltatemp(2)      &
           &           +                   tables(ptr+10)*3.0d0*deltatemp(3))     &
           &           + 2.0d0*deltapres(2)*(tables(ptr+14)*1.0d0*deltatemp(1)    &
           &           +                   tables(ptr+15)*2.0d0*deltatemp(2)      &
           &           +                   tables(ptr+16)*3.0d0*deltatemp(3))     &
           &           + 3.0d0*deltapres(3)*(tables(ptr+20)*1.0d0*deltatemp(1)    &
           &           +                   tables(ptr+21)*2.0d0*deltatemp(2)      &
           &           +                   tables(ptr+22)*3.0d0*deltatemp(3))     &
           &           + 4.0d0*deltapres(4)*(tables(ptr+26)*1.0d0*deltatemp(1)    &
           &           +                   tables(ptr+27)*2.0d0*deltatemp(2)      &
           &           +                   tables(ptr+28)*3.0d0*deltatemp(3))     &
           &           + 5.0d0*deltapres(5)*(tables(ptr+32)*1.0d0*deltatemp(1)    &
           &           +                   tables(ptr+33)*2.0d0*deltatemp(2)      &
           &           +                   tables(ptr+34)*3.0d0*deltatemp(3))
!
         RETURN
      END FUNCTION gptfuncte
!#################################################
      REAL(sdkx) FUNCTION gptfunctline(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gpt(d2g/dpdt) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
         gptfunctline =                                                        &
       &                1.0d0*deltapres(1)*(tables(ptr+ 8)*1.0d0*deltatemp(1)  &
       &              +                     tables(ptr+ 9)*2.0d0*deltatemp(2)) &
       &              + 2.0d0*deltapres(2)*(tables(ptr+14)*1.0d0*deltatemp(1)  &
       &              +                     tables(ptr+15)*2.0d0*deltatemp(2))
!
         RETURN
      END FUNCTION gptfunctline
!#########################################3
      REAL(sdkx) FUNCTION gtfunct(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gt(dg/dt) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
!
             gtfunct =                                                      &
           &           deltapres(1)*(tables(ptr+ 2)*1.0d0*deltatemp(1)      &
           &         +               tables(ptr+ 3)*2.0d0*deltatemp(2)      &
           &         +               tables(ptr+ 4)*3.0d0*deltatemp(3)      &
           &         +               tables(ptr+ 5)*4.0d0*deltatemp(4)      &
           &         +               tables(ptr+ 6)*5.0d0*deltatemp(5))     &
           &         + deltapres(2)*(tables(ptr+ 8)*1.0d0*deltatemp(1)      &
           &         +               tables(ptr+ 9)*2.0d0*deltatemp(2)      &
           &         +               tables(ptr+10)*3.0d0*deltatemp(3)      &
           &         +               tables(ptr+11)*4.0d0*deltatemp(4)      &
           &         +               tables(ptr+12)*5.0d0*deltatemp(5))     &
           &         + deltapres(3)*(tables(ptr+14)*1.0d0*deltatemp(1)      &
           &         +               tables(ptr+15)*2.0d0*deltatemp(2)      &
           &         +               tables(ptr+16)*3.0d0*deltatemp(3)      &
           &         +               tables(ptr+17)*4.0d0*deltatemp(4)      &
           &         +               tables(ptr+18)*5.0d0*deltatemp(5))
             gtfunct = gtfunct                                              &
           &         + deltapres(4)*(tables(ptr+20)*1.0d0*deltatemp(1)      &
           &         +               tables(ptr+21)*2.0d0*deltatemp(2)      &
           &         +               tables(ptr+22)*3.0d0*deltatemp(3)      &
           &         +               tables(ptr+23)*4.0d0*deltatemp(4)      &
           &         +               tables(ptr+24)*5.0d0*deltatemp(5))     &
           &         + deltapres(5)*(tables(ptr+26)*1.0d0*deltatemp(1)      &
           &         +               tables(ptr+27)*2.0d0*deltatemp(2)      &
           &         +               tables(ptr+28)*3.0d0*deltatemp(3)      &
           &         +               tables(ptr+29)*4.0d0*deltatemp(4)      &
           &         +               tables(ptr+30)*5.0d0*deltatemp(5))     &
           &         + deltapres(6)*(tables(ptr+32)*1.0d0*deltatemp(1)      &
           &         +               tables(ptr+33)*2.0d0*deltatemp(2)      &
           &         +               tables(ptr+34)*3.0d0*deltatemp(3)      &
           &         +               tables(ptr+35)*4.0d0*deltatemp(4)      &
           &         +               tables(ptr+36)*5.0d0*deltatemp(5))
         RETURN
      END FUNCTION gtfunct
!########################################
      REAL(sdkx) FUNCTION gtfuncte(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gt(dg/dt) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
!
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
!
             gtfuncte =                                                       &
           &            deltapres(1)*(tables(ptr+ 2)*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+ 3)*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+ 4)*3.0d0*deltatemp(3))      &
           &          + deltapres(2)*(tables(ptr+ 8)*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+ 9)*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+10)*3.0d0*deltatemp(3))      &
           &          + deltapres(3)*(tables(ptr+14)*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+15)*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+16)*3.0d0*deltatemp(3))
             gtfuncte = gtfuncte                                              &
           &          + deltapres(4)*(tables(ptr+20)*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+21)*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+22)*3.0d0*deltatemp(3))      &
           &          + deltapres(5)*(tables(ptr+26)*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+27)*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+28)*3.0d0*deltatemp(3))      &
           &          + deltapres(6)*(tables(ptr+32)*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+33)*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+34)*3.0d0*deltatemp(3))
         RETURN
      END FUNCTION gtfuncte
!###########################################################
      REAL(sdkx) FUNCTION gttfunct(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gtt(d2g/dt2) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
             gttfunct = deltapres(1)*(tables(ptr+ 3)*2.0d0*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+ 4)*3.0d0*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+ 5)*4.0d0*3.0d0*deltatemp(3)       &
           &          +               tables(ptr+ 6)*5.0d0*4.0d0*deltatemp(4))      &
           &          + deltapres(2)*(tables(ptr+ 9)*2.0d0*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+10)*3.0d0*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+11)*4.0d0*3.0d0*deltatemp(3)       &
           &          +               tables(ptr+12)*5.0d0*4.0d0*deltatemp(4))      &
           &          + deltapres(3)*(tables(ptr+15)*2.0d0*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+16)*3.0d0*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+17)*4.0d0*3.0d0*deltatemp(3)       &
           &          +               tables(ptr+18)*5.0d0*4.0d0*deltatemp(4))
             gttfunct = gttfunct                                                    &
           &          + deltapres(4)*(tables(ptr+21)*2.0d0*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+22)*3.0d0*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+23)*4.0d0*3.0d0*deltatemp(3)       &
           &          +               tables(ptr+24)*5.0d0*4.0d0*deltatemp(4))      &
           &          + deltapres(5)*(tables(ptr+27)*2.0d0*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+28)*3.0d0*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+29)*4.0d0*3.0d0*deltatemp(3)       &
           &          +               tables(ptr+30)*5.0d0*4.0d0*deltatemp(4))      &
           &          + deltapres(6)*(tables(ptr+33)*2.0d0*1.0d0*deltatemp(1)       &
           &          +               tables(ptr+34)*3.0d0*2.0d0*deltatemp(2)       &
           &          +               tables(ptr+35)*4.0d0*3.0d0*deltatemp(3)       &
           &          +               tables(ptr+36)*5.0d0*4.0d0*deltatemp(4))
         RETURN
      END FUNCTION gttfunct
!###########################################################
      REAL(sdkx) FUNCTION gttfuncte(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gtt(d2g/dt2) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
             gttfuncte = deltapres(1)*(tables(ptr+ 3)*2.0d0*1.0d0*deltatemp(1)      &
           &           +               tables(ptr+ 4)*3.0d0*2.0d0*deltatemp(2))     &
           &           + deltapres(2)*(tables(ptr+ 9)*2.0d0*1.0d0*deltatemp(1)      &
           &           +               tables(ptr+10)*3.0d0*2.0d0*deltatemp(2))     &
           &           + deltapres(3)*(tables(ptr+15)*2.0d0*1.0d0*deltatemp(1)      &
           &           +               tables(ptr+16)*3.0d0*2.0d0*deltatemp(2))
             gttfuncte = gttfuncte                                                  &
           &           + deltapres(4)*(tables(ptr+21)*2.0d0*1.0d0*deltatemp(1)      &
           &           +               tables(ptr+22)*3.0d0*2.0d0*deltatemp(2))     &
           &           + deltapres(5)*(tables(ptr+27)*2.0d0*1.0d0*deltatemp(1)      &
           &           +               tables(ptr+28)*3.0d0*2.0d0*deltatemp(2))     &
           &           + deltapres(6)*(tables(ptr+33)*2.0d0*1.0d0*deltatemp(1)      &
           &           +               tables(ptr+34)*3.0d0*2.0d0*deltatemp(2))
         RETURN
      END FUNCTION gttfuncte
!###########################################################
      REAL(sdkx) FUNCTION gttfunctline(deltapres, deltatemp, tables, ptr)
!
!  Evaluate gtt(d2g/dt2) as a function of dimensionless
!    deltapres and deltatemp given the
!    coefficients of the bi-quintic polynomial.
!  coefficients of the bi-quintic polynomial.
!  You can pass in a 1-D vector coeff as long as the first index is for
!  The row index(first index) for c(ip,jt) corresponds to the deltapres
!  The column index(second index) for c(ip,jt) corresponds to the delta
!
         IMPLICIT NONE
         REAL(sdk) :: tables(:)
         REAL(sdk) :: deltapres(:), deltatemp(:)
         INTEGER(sik) :: ptr
!
         gttfunctline =                                             &
       &           deltapres(1)*(tables(ptr+ 3)*2.0d0*deltatemp(1)) &
       &         + deltapres(2)*(tables(ptr+ 9)*2.0d0*deltatemp(1)) &
       &         + deltapres(3)*(tables(ptr+15)*2.0d0*deltatemp(1))
!
         RETURN
      END FUNCTION gttfunctline
!#############################################      
              
END MODULE EosIAPWSCrunch