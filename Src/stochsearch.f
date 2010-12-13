c     Used in PyMCMC for the stochastic search algorithm.    
c     Copyright (C) 2010  Chris Strickland
c
c     This program is free software: you can redistribute it and/or modify
c     it under the terms of the GNU General Public License as published by
c     the Free Software Foundation, either version 3 of the License, or
c     (at your option) any later version.
c
c     This program is distributed in the hope that it will be useful,
c     but WITHOUT ANY WARRANTY; without even the implied warranty of
c     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c     GNU General Public License for more details.

c     You should have received a copy of the GNU General Public License
c     along with this program.  If not, see <http://www.gnu.org/licenses/>. 


c     ssreg is a Fortran77 subroutine that is used by the Python package
c     PyMCMC. Specifically ssreg is used to calculate the indicator
c     vector that is used in the Bayesian stochastic search variable
c     selection problem for the normal linear model. The subroutine
c     assumes the the model is specified using a g-prior. If y denotes
c     the observation vector and x denonotes the set of regressors then
c     ypy = y'y
c     g = g from the g-prior
c     xpx = x'x
c     spy = x'y
c     xgxg = a (k,k) matrix used for workings
c     xgy = a (k,1) vector used for workings
c     gam = a (k,1) vector of integers used to store the selection
c           dummies
c     ru = a (k,1) vector of uniform random numbers     
c     w = a (k,6) matrix for workings
c     w2 = a (k,k) matrix for workings
c     ifo = an integer. Equals 0 on return if computuations were
c           successful
c     ifo2 = an integer. Equals 0 on return if computuations were
c            successful
c     n = the number of observations
c     k = the number of regressors
      
      subroutine ssreg(ypy,g,xpx,xpy,xgxg,xgy,gam,ru,w,w2,ifo, ifo2,n,k)
      implicit none
      integer n,k,gam(k),ifo, ifo2
      real*8 xpy(k), g, xpx(k,k), xgy(k),xgxg(k,k),ru(k),ypy
      real*8 w(k,6),w2(k,k), pgamnum,pgamdenom
      integer qgam, sumgam,i
      real*8 probdenom

cf2py intent(inout) gam
cf2py intent(in) ypy
cf2py intent(in) g
cf2py intent(in) xpx
cf2py intent(in) xgxg
cf2py intent(in) w2
cf2py intent(in) xgq
cf2py intent(in) ru
cf2py intent(in) n
cf2py intent(in) k
cf2py intent(inout) ifo
cf2py intent(inout) ifo2
      do i=2,k
          qgam=sumgam(gam,k)      
          if (gam(i).eq.1) then
              call precalc(qgam,xpy,xpx,xgy,xgxg,w2,gam,w,k,ifo2)
              call probgam(pgamnum,g,ypy,qgam,xgxg,w2,xgy,w,n,k,ifo)
              gam(i)=0
              qgam=qgam-1
              call precalc(qgam,xpy,xpx,xgy,xgxg,w2,gam,w,k,ifo2)
              call probgam(pgamdenom,g,ypy,qgam,xgxg,w2,xgy,w,n,k,ifo)
              pgamdenom=probdenom(pgamnum,pgamdenom)
              if (ru(i)>pgamdenom.or.ifo2.ne.0.or.ifo.ne.0) then
                  gam(i)=1
                  qgam=qgam+1
              endif
          else
              call precalc(qgam,xpy,xpx,xgy,xgxg,w2,gam,w,k,ifo2)
              call probgam(pgamnum,g,ypy,qgam,xgxg,w2,xgy,w,n,k,ifo)
              gam(i)=1
              qgam=qgam+1
              call precalc(qgam,xpy,xpx,xgy,xgxg,w2,gam,w,k,ifo2)
              call probgam(pgamdenom,g,ypy,qgam,xgxg,w2,xgy,w,n,k,ifo)
              pgamdenom=probdenom(pgamnum,pgamdenom)
              if (ru(i)>pgamdenom.or.ifo2.ne.0.or.ifo.ne.0) then
                  gam(i)=0
                  qgam=qgam-1
              endif
          endif
      enddo
      end

c     subroutine that does precalulation for probgam
      subroutine precalc(qgam,xpy,xpx,xgy,xgxg,w2,gam,w,k,ifo2)
      implicit none
      integer k, ifo2,gam(k),qgam
      real*8 xpy(k), xpx(k,k),w2(k,k),xgy(k),xgxg(k,k),w(k,6)

      call calcxpy(xpy,gam,xgy,k)
      call calcxpx(xpx,gam,xgxg,k)
c     calculate prior mean
      call calcpm(qgam,xgxg,w2,gam,w,k,ifo2)
      end
      
c     subroutine calculates the prior mean
      subroutine calcpm(qgam,xgxg,w2,gam,w,k,ifo2)
      implicit none
      integer k,i,j,ifo2,qgam
      real*8  xgxg(k,k),w2(k,k), gam(k), w(k,6) 

      j=0
      do i=1,k
          if (gam(i).eq.1) then
              w(j,5)=w(i,4)
              j=j+1
          endif
      enddo
      do j=1,qgam
          w(j,1)=w(j,5)
          do i=1,qgam
              w2(i,j)=xgxg(i,j)
          enddo
      enddo
      call dposv('u',qgam,1,w2,k,w(:,1),k,ifo2)
      end  
        
      


c     subroutine calculates the probability of gamma      
      subroutine probgam(pgam,g,ypy,qgam,xgxg,w2,xgy,w,n,k,info)
      implicit none
      integer n, k, qgam,info, i,j
      real*8 pgam, xgxg(k,k),w2(k,k), xgy(k), w(k,6),c1, c2,c3,g, ypy
      real*8 alpha,beta
c     BLAS function declation
      real*8 ddot


      c1=(-dble(qgam+1)/2.0)*log(g+1.0)
      do j=1,qgam
          w(j,2)=xgy(j)
          do i=1,qgam
              w2(i,j)=xgxg(i,j)
          enddo
      enddo
      alpha=1.0
      beta=0.0
      call dposv('u',qgam,1,w2,k,w(:,2),k,info)
      if(info.eq.0) then
          c2=g/(g+1)*ddot(qgam,xgy,1,w(:,2),1)
          
       call dgemv('n',qgam,qgam,alpha,xgxg,k,w(:,1),1,beta,w(:,3),1)
          c3=ddot(qgam,w(:,1),1,w(:,3),1)/(g+1.0)

          pgam=c1+(-(dble(n)/2.0))*log(ypy-c2-c3)
      else
          pgam=-1.0D256
      endif
      end

c     function returns the probability of the denominator
c     lnp1 is the unnormalised log probability of the numerator, while
c     lnp2 is the unnormalised log probability of the denonator
      real*8 function probdenom(lnp1,lnp2)
      real*8 lnp2,lnp1,maxp,lnsump

      maxp=max(lnp1,lnp2)
      lnsump=log(exp(lnp1-maxp)+exp(lnp2-maxp))+maxp
      probdenom=exp(lnp2-lnsump)
      return
      end

      
c     functions calulates the number of ones in gam      
      integer function sumgam(gam,k)
      integer k, gam(k),i

      sumgam=0
      do i=1,k
          sumgam=sumgam+gam(i)
      enddo
      return
      end    

c     function extracts the rows in xpy that correspond to ones in gam
      subroutine calcxpy(xpy,gam,xgy,k)
      implicit none
      integer k,gam(k), i,j
      real*8 xpy(k), xgy(k)

      j=1
      do i=1,k
          if(gam(i).eq.1) then
              xgy(j)=xpy(i)
              j=j+1
          endif
      enddo
      end

      subroutine calcxpx(xpx,gam,xgxg,k)
      implicit none
      integer k, gam(k),i,j,s,t
      real*8 xpx(k,k),xgxg(k,k) 

      s=1
      t=1
      do j=1,k
          do i=1,k
              if(gam(i).eq.1 .and. gam(j).eq.1) then
                  xgxg(s,t)=xpx(i,j)
                  s=s+1
              endif
          enddo
          if(gam(j).eq.1) then
              t=t+1
          endif
          s=1
      enddo
      end

c     functions for the normal inverted gamma prior
     
c     subroutine samples gamma for normal-inverted gamma prior
      subroutine ssreg_nig(ypy,ldr,vs,vxy,vobar,vubar,gam,xpx,xpy,v,r,
     + nuo,ru,k)

      implicit none
      integer k,gam(k),gami,ifo,ifo2,nuo,i
      real*8 vobar(k,k),vubar(k,k),xpx(k,k),xpy(k),v(k,2),r(k,k)
      real*8 vxy(k),ldr,ypy,vs
      real*8 pgamnum,pgamdenom,ru(k),probdenom,marg

cf2py intent(in) ypy
cf2py intent(in) ldr
cf2py intent(in) vs
cf2py intent(in) vxy
cf2py intent(in) vobar
cf2py intent(in) vubar
cf2py intent(in) gam
cf2py intent(in) xpx
cf2py intent(in) xpy
cf2py intent(in) v
cf2py intent(in) r
cf2py intent(in) gam
cf2py intent(in) nuo
cf2py intent(in) ru
cf2py intent(in) k

      do i=2,k
          if (gam(i).eq.1) then
              gami=-1
              pgamnum=marg(ypy,ldr,vs,vxy,vobar,vubar,gam,xpx,xpy,v,
     + r,gami,ifo,nuo,k)
              gam(i)=0
              gami=i
              pgamdenom=marg(ypy,ldr,vs,vxy,vobar,vubar,gam,xpx,xpy,v,
     + r,gami,ifo2,nuo,k)
              pgamdenom=probdenom(pgamnum,pgamdenom)
              if (ru(i)>pgamdenom.or.ifo2.ne.0.or.ifo.ne.0) then
                  gam(i)=1
                  call update_vubar(vubar,gam,v,r,gami,k)
              endif
          else
              gami=-1
              pgamnum=marg(ypy,ldr,vs,vxy,vobar,vubar,gam,xpx,xpy,v,
     + r,gami,ifo,nuo,k)
              gam(i)=1
              gami=i
              pgamdenom=marg(ypy,ldr,vs,vxy,vobar,vubar,gam,xpx,xpy,v,
     + r,gami,ifo2,nuo,k)
              pgamdenom=probdenom(pgamnum,pgamdenom)
              if (ru(i)>pgamdenom.or.ifo2.ne.0.or.ifo.ne.0) then
                  gam(i)=0
                  call update_vubar(vubar,gam,v,r,gami,k)
              endif
          endif
      enddo
      end


      

c     subroutine to initialise vubar
      subroutine initialise_vubar(vubar,gam,v,r,k)
      implicit none
      integer k,gam(k),i,j
      real*8 vubar(k,k),r(k,k),v(k,2)

cf2py intent(inout) vubar
cf2py intent(in) gam
cf2py intent(in) v
cf2py intent(in) r
cf2py intent(in) k

      do j=1,k
          do i=1,k
              vubar(i,j)=v(i,gam(i)+1)*r(i,j)*v(j,gam(j)+1)
          enddo
      enddo
      end



c     subroutine updates vubar for a change in gamma
      subroutine update_vubar(vubar,gam,v,r,gami,k)
      implicit none
      integer k,gam(k),gami,i,j
      real*8 vubar(k,k),r(k,k),v(k,2),tmp

      do j=1,k
          vubar(gami,j)=v(gami,gam(gami)+1)*r(gami,j)
      enddo
      do i=1,k
          tmp=v(gami,gam(gami)+1)*vubar(i,gami)
          vubar(i,gami)=tmp
      enddo
      end


c     subroutine calculates marginal likelihood
      real*8 function marg(ypy,ldr,vs,vxy,vobar,vubar,gam,xpx,xpy,v,r,
     + gami,ifo,nuo,k)
      implicit none
      integer k,gam(k),gami,ifo,nuo,i
      real*8 vobar(k,k),vubar(k,k),xpx(k,k),xpy(k),v(k,2),r(k,k)
      real*8 alpha,beta,vxy(k),ddot,vso,ldr,lndetvu,ypy,vs
      real*8 lndetvo,logdetvu
        
      if (gami.ne.-1) then
          call update_vubar(vubar,gam,v,r,gami,k)
      endif
      alpha=1.0
      do i=1,k
          call dcopy(k,vubar(:,i),1,vobar(:,i),1)
          call daxpy(k,alpha,xpx(:,i),1,vobar(:,i),1)
      enddo

      call dpotrf('u',k,vobar,k,ifo)
      if (ifo.eq.0) then
          call dcopy(k,xpy,1,vxy,1)
          call dtrsv('u','t','n',k,vobar,k,vxy,1)
          beta=0.0
          vso=vs+ypy-ddot(k,vxy,1,vxy,1)

          lndetvo=0.0
          do i=1,k
              lndetvo=lndetvo+log(vobar(i,i))
          enddo

          marg=0.5*logdetvu(v,gam,ldr,k)-lndetvo-dble(nuo)/2.0*
     + log(vso/2.0)
      else
          marg=-1.0D256
      endif
      return
      end
c     function returns the log determinant of vubar
      real*8 function logdetvu(v,gam,ldr,k)
      implicit none
      integer i,k,gam(k)
      real*8 v(k,k),ldr


      logdetvu=0.0
      do i=1,k
          logdetvu=logdetvu+log(v(i,gam(i)+1))
      enddo
      logdetvu=2.0*logdetvu+ldr
      return
      end
      

c     Stochastic search gamma|beta, sigma normal-inverted gamma prior

c     subroutine to sample gamma

      subroutine ssregcbeta_nig(beta,sig,vub,ldr,vubar,gam,v,r,ru,k)
      implicit none
      integer k,gam(k),gami,i
      real*8 v(k,2),r(k,k),beta(k),sig,vub(k),vubar(k,k)
      real*8 ldr,pgamnum,pgamdenom,ru(k),probdenom,probgambet

cf2py intent(in) beta
cf2py intent(in) sig
cf2py intent(in) vub
cf2py intent(in) ldr
cf2py intent(in) vubar
cf2py intent(in) gam
cf2py intent(in) v
cf2py intent(in) r
cf2py intent(in) gam
cf2py intent(in) ru
cf2py intent(in) k

      do i=2,k
          if (gam(i).eq.1) then
              gami=-1
              pgamnum=probgambet(ldr,beta,vub,sig,vubar,v,r,gami,gam,k)
              gam(i)=0
              gami=i
              pgamdenom=probgambet(ldr,beta,vub,sig,vubar,v,r,gami,gam,
     + k)
              pgamdenom=probdenom(pgamnum,pgamdenom)
              if (ru(i)>pgamdenom) then
                  gam(i)=1
                  call update_vubar(vubar,gam,v,r,gami,k)
              endif
          else
              gami=-1
              pgamnum=probgambet(ldr,beta,vub,sig,vubar,v,r,gami,gam,k)
              gam(i)=1
              gami=i
              pgamdenom=probgambet(ldr,beta,vub,sig,vubar,v,r,gami,gam,
     + k)
              pgamdenom=probdenom(pgamnum,pgamdenom)
              if (ru(i)>pgamdenom) then
                  gam(i)=0
                  call update_vubar(vubar,gam,v,r,gami,k)
              endif
          endif
      enddo
      end


c     subroutine calculate prob beta,gamma
      real*8 function probgambet(ldr,beta,vub,sig,vubar,v,r,gami,gam,k)
      implicit none
      integer k,gam(k),gami
      real*8 beta(k),v(k,2),r(k,k),vubar(k,k),ldr,logdetvu
      real*8 vub(k),alpha,beta1,sig,ddot

      if (gami.ne.-1) then
          call update_vubar(vubar,gam,v,r,gami,k)
      endif

      alpha=1.0
      beta1=0.0
      call dgemv('n',k,k,alpha,vubar,k,beta,1,beta1,vub,1)
      probgambet=0.5*logdetvu(v,gam,ldr,k)-0.5/sig**2
     + *ddot(k,beta,1,vub,1)
      return
      end

      


