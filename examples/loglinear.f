c     fortran 77 code used to calculate the likelihood of a log
c     linear model. Function uses BLAS.

      subroutine logl(xb,xm,bv,yv,llike,n,k)
      implicit none
      integer n, k, i
      real*8 xb(n),xm(n,k), bv(k), yv(n), llike
      real*8 alpha, beta

cf2py intent(in,out) llike
cf2py intent(in) yv
cf2py intent(ini bv
cf2py intent(in) xmat 
cf2py intent(in) xb 

      alpha=1.0
      beta=0.0
      call dgemv('n',n,k,alpha,xm,n,bv,1,beta,xb,1)

      llike=0.0
      do i=1,n
          llike=llike+yv(i)*xb(i)-exp(xb(i))
      enddo
      end

          
      


