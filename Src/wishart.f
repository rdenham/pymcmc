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


c     subroutine constructs the upper Cholesky triangle used to simulate
c     simulate from the cholesky decomposition. See Strickland et al
c     2009. This subroutine is called from regtools.py in PyMCMC.

      subroutine chol_wishart(rn,rc,u,c,r,p)
      implicit none
      integer p,i,j,k
      real*8 rc(p),rn(p*(p-1)/2),u(p,p),c(p,p),r(p,p)
      real*8 alpha, beta

cf2py intent(in) rc
cf2py intent(in) rn
cf2py intent(in) u
cf2py intent(in) c
cf2py intent(inout) r
cf2py intent(in) p

      k=1
      do i=1,p
          u(i,i)=sqrt(rc(i))
          do j=i+1,p
              u(i,j)=rn(k)
              k=k+1
          enddo
          do j=1,i-1
              u(i,j)=0.0
          enddo
      enddo

      alpha=1.0
      beta=0.0

      call dgemm('n','n',p,p,p,alpha,c,p,u,p,beta,r,p)
      end

c     routine used to sample from the wishart distribution
      subroutine chol_wishart2(rn,rc,u,uc,so,ifo,p)
      implicit none
      integer p,i,j,k,ifo
      real*8 rc(p),rn(p*(p-1)/2),u(p,p),so(p,p)
      real*8 uc(p,p)
      real*8 alpha

cf2py intent(in) rc
cf2py intent(in) rn
cf2py intent(inout) u
cf2py intent(in) uc
cf2py intent(in) c
cf2py intent(in) so
cf2py intent(in) p

      k=1
      do i=1,p
          u(i,i)=sqrt(rc(i))
          uc(i,i)=u(i,i)
          do j=i+1,p
              u(i,j)=rn(k)
              k=k+1
          enddo
          do j=1,i-1
              u(i,j)=0.0
          enddo
      enddo

      do j=1,p
          call dcopy(p,u(:,j),1,uc(:,j),1)
      enddo

      call dpotrf('u',p,so,p,ifo)
      alpha=1.0
      call dtrsm('l','u','t','n',p,p,alpha,so,p,u,p)
      call dtrsm('l','u','n','n',p,p,alpha,so,p,u,p)
      call dtrmm('l','u','t','n',p,p,alpha,uc,p,u,p)
      end




      
c     subroutine constructs second shape parameter for 
c     wishart distribution from residuals
      subroutine calc_sobar(su,so,res,p,n)
      implicit none
      integer n,p,t,i
      real*8 su(p,p),so(p,p),res(n,p)
      real*8 alpha

cf2py intent(in) su
cf2py intent(inout) so
cf2py intent(in) res
cf2py intent(in) p
cf2py intent(in) n

      do i=1,p
          call dcopy(p,su(:,i),1,so(:,i),1)
      enddo

      alpha=1.0
      do t=1,n
          call dger(p,p,alpha,res(t,:),1,res(t,:),1,so,p)
      enddo
      end
          



