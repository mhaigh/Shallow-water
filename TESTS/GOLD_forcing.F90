program GOLD_forcing    

integer :: js,je,is,ie
integer :: jSo, jNo, j2
integer :: out_unit
real :: Amp
real, dimension(256) :: buoy

out_unit = 1

Amp = 1
js = 1
je = 256

j2 = int((je - js) / 2) + js ; jN = je - js + 1
jSo = int((j2 - js) / 2) + js ; jNo = int((je - j2) / 2) + j2 

do j=js,jSo ; do i=is,ie
  buoy(j) = Amp * (j -js) &
                        / (jSo - js)
enddo ; enddo 
do j=jSo+1,jNo ; do i=is,ie
  buoy(j) = Amp * (2 * (jNo - j) - (jNo - jSo)) &
                         / (jNo - jSo)
enddo ; enddo
do j=jNo+1,je ; do i=is,ie
  buoy(j) = Amp * (j - je) &
                       / (je - jNo)
enddo ; enddo

open (unit=out_unit,file="results.txt",action="write",status="replace")
do j=js,je
  write (out_unit,*) buoy(j)
enddo



end program GOLD_forcing
