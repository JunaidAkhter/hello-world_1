"""
doApplyHam(psiIn,hloc,N,usePBC)
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 06/2020
------------------------
Applies local Hamiltonian (given as sum of nearest neighbor terms 'hloc')
to input state 'psiIn'. Number of lattice sites specified as 'N' while
'usePBC' determines whether open or periodic boundaries are used.
"""
function doApplyHam(psiIn,hloc,N,usePBC)

  d = size(hloc,1);
  psiOut = zeros(d^N,1);
  for k = 1:N-1
    # apply local Hamiltonian terms to sites [k,k+1]
    cont_inds = [[2],[2]];
    psi_temp = tensordot(reshape(hloc,d^2,d^2),
      reshape(psiIn,d^(k-1), d^2, d^(N-1-k)),cont_inds);
    psiOut = psiOut + reshape(permutedims(psi_temp,[2,1,3]),d^N);
  end

  if usePBC
    # apply periodic term
    cont_inds = [[3,4],[3,1]];
    psi_temp = tensordot(reshape(hloc,d,d,d,d),
      reshape(psiIn,d, d^(N-2), d),cont_inds);
    psiOut = psiOut + reshape(permutedims(psi_temp,[2,3,1]),d^N);
  end

  return psiOut
end


"""
tensordot(A,B,cont_inds):
Function for taking the produce of two tensors A and B, designed to mimic the
numpy tensordot. Input `cont_inds = [A_axes,B_axes]` where `A_axes` and
`B_axes` are vectors describing the indices to be contracted, e.g. set
cont_inds = [[2],[1]] to contract the 2nd index of A with the 1st index of B.
"""
function tensordot(A, B, cont_inds)

  A_free = deleteat!(collect(1:ndims(A)), sort(cont_inds[1]))
  B_free = deleteat!(collect(1:ndims(B)), sort(cont_inds[2]))
  A_perm = vcat(A_free, cont_inds[1])
  B_perm = vcat(cont_inds[2], B_free)

  return reshape(
    reshape(
      permutedims(A, A_perm),
      prod(size(A)[A_free]),
      prod(size(A)[cont_inds[1]]),
    ) * reshape(
      permutedims(B, B_perm),
      prod(size(B)[cont_inds[2]]),
      prod(size(B)[B_free]),
    ),
    (size(A)[A_free]..., size(B)[B_free]...),
  )
end
