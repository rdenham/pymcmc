c     Function calculates the autocorrelation function
c     Copyright (C) 2010 Chris Strickland

c     This program is free software: you can redistribute it and/or modify
c     it under the terms of the GNU General Public License as published by
c     the Free Software Foundation, either version 3 of the License, or
c     (at your option) any later version.

c     This program is distributed in the hope that it will be useful,
c     but WITHOUT ANY WARRANTY; without even the implied warranty of
c     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c     GNU General Public License for more details.

c     You should have received a copy of the GNU General Public License
c     along with this program.  If not, see <http://www.gnu.org/licenses/>.


c     acf is a Fortran77 subroutine for calculating the autocorrelation 
c     function. The computation is done in the time domain. 

c     ts = the timeseries of interest.
c     n = the number of observations.
c     corr =  a vector of length nlag that stores the autocorrelation
c             function for the timeseries of interest on exit.
c     nlag = the number of lags to be used in the calculation of the
c            autocorrelation function.

      subroutine acf(ts,n,corr,nlag)
      integer n,nlag
      double precision ts(n), corr(nlag),mean, var

cf2py intent(in) ts
cf2py intent(in,out) corr
cf2py intent(in) n
cf2py intent(in) nlag
cf2py depend(ts) n
cf2py depend(corr) nlag


      mean=0.0
      do i=1,n
          mean=mean+ts(i)
      enddo

      mean=mean/dble(n)
      

      do i=1,nlag
          corr(i)=0.0
          do j=1,n-i
              corr(i)=corr(i)+(ts(j)-mean)*(ts(j+i-1)-mean)
          enddo
      enddo
      var=corr(1)/dble(n)
    
      do i=1,nlag
          corr(i)=corr(i)/(var*dble(n))
      enddo    
      end
c     end of subroutine acf






      


      

      



