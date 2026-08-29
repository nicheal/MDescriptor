module mdescriptor_soap_turbo_reference
  use iso_c_binding, only: c_double, c_int
  use soap_turbo_desc, only: get_soap
  use soap_turbo_compress_module, only: get_compress_indices
  implicit none

contains

  subroutine soap_turbo_reference( &
      n_sites, n_species, n_atom_pairs, l_max, feature_count, int_data, real_data, basis_mode, compression_mode, soap_out) &
      bind(C, name="soap_turbo_reference")
    integer(c_int), value, intent(in) :: n_sites, n_species, n_atom_pairs, l_max, feature_count
    integer(c_int), intent(in) :: int_data(*)
    real(c_double), intent(in) :: real_data(*)
    integer(c_int), value, intent(in) :: basis_mode, compression_mode
    real(c_double), intent(out) :: soap_out(*)

    integer :: i, j, k, p, max_neighbors, dense_count, compression_dim, compression_nonzero
    integer :: n_sites_f, n_species_f, n_atom_pairs_f, l_max_f, feature_count_f, radial_enhancement_f
    integer, allocatable :: species(:,:), species_multiplicity(:), compress_i(:), compress_j(:)
    integer, allocatable :: n_neigh(:), alpha_max(:)
    logical, allocatable :: mask(:,:)
    logical :: compress_soap_f
    real(8), allocatable :: soap(:,:), soap_cart_der(:,:,:), compress_el(:)
    real(8), allocatable :: rjs(:), thetas(:), phis(:), rcut_hard(:), rcut_soft(:), nf(:)
    real(8), allocatable :: global_scaling(:), atom_sigma_r(:), atom_sigma_r_scaling(:)
    real(8), allocatable :: atom_sigma_t(:), atom_sigma_t_scaling(:), amplitude_scaling(:), central_weight(:)
    character(len=16) :: basis, scaling_mode, compress_mode_f

    n_sites_f = n_sites
    n_species_f = n_species
    n_atom_pairs_f = n_atom_pairs
    l_max_f = l_max
    feature_count_f = feature_count
    radial_enhancement_f = 0
    max_neighbors = max(1, n_atom_pairs_f)
    basis = "poly3"
    if (basis_mode == 1) basis = "poly3gauss"
    scaling_mode = "polynomial"

    allocate(species(1, n_sites_f), species_multiplicity(n_sites_f), n_neigh(n_sites_f), alpha_max(n_species_f))
    p = 1
    do i = 1, n_sites_f
      species(1, i) = int_data(p)
      p = p + 1
    end do
    species_multiplicity = 1
    allocate(mask(max_neighbors, n_species_f))
    mask = .false.
    do i = 1, n_atom_pairs_f
      mask(i, int_data(p)) = .true.
      p = p + 1
    end do
    do i = 1, n_sites_f
      n_neigh(i) = int_data(p)
      p = p + 1
    end do
    do i = 1, n_species_f
      alpha_max(i) = int_data(p)
      p = p + 1
    end do

    allocate(rjs(n_atom_pairs_f), thetas(n_atom_pairs_f), phis(n_atom_pairs_f))
    allocate(rcut_hard(n_species_f), rcut_soft(n_species_f), nf(n_species_f), global_scaling(n_species_f))
    allocate(atom_sigma_r(n_species_f), atom_sigma_r_scaling(n_species_f))
    allocate(atom_sigma_t(n_species_f), atom_sigma_t_scaling(n_species_f))
    allocate(amplitude_scaling(n_species_f), central_weight(n_species_f))
    p = 1
    do i = 1, n_atom_pairs_f
      rjs(i) = real_data(p)
      p = p + 1
    end do
    do i = 1, n_atom_pairs_f
      thetas(i) = real_data(p)
      p = p + 1
    end do
    do i = 1, n_atom_pairs_f
      phis(i) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      rcut_hard(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      rcut_soft(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      nf(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      global_scaling(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      atom_sigma_r(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      atom_sigma_r_scaling(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      atom_sigma_t(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      atom_sigma_t_scaling(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      amplitude_scaling(j) = real_data(p)
      p = p + 1
    end do
    do j = 1, n_species_f
      central_weight(j) = real_data(p)
      p = p + 1
    end do

    dense_count = sum(alpha_max) * (sum(alpha_max) + 1) / 2 * (l_max_f + 1)
    compress_soap_f = .false.
    compress_mode_f = ""
    select case (compression_mode)
    case (1)
      compress_soap_f = .true.
      compress_mode_f = "trivial"
    case (2:10)
      compress_soap_f = .true.
      i = compression_mode - 2
      compress_mode_f = char(ichar("0") + i / 3) // "_" // char(ichar("0") + mod(i, 3))
    end select
    compression_dim = dense_count
    compression_nonzero = 0
    allocate(compress_i(max(1, 2 * dense_count)), compress_j(max(1, 2 * dense_count)), &
             compress_el(max(1, 2 * dense_count)))
    compress_i = 1
    compress_j = 1
    compress_el = 0.d0
    if (compress_soap_f) then
      call get_compress_indices(trim(compress_mode_f), alpha_max, l_max_f, compression_dim, compression_nonzero, &
                                compress_i, compress_j, compress_el, "get_dim")
      call get_compress_indices(trim(compress_mode_f), alpha_max, l_max_f, compression_dim, compression_nonzero, &
                                compress_i, compress_j, compress_el, "set_indices")
    end if
    allocate(soap(feature_count_f, n_sites_f), soap_cart_der(3, max(1, feature_count_f), max_neighbors))
    soap = 0.d0
    soap_cart_der = 0.d0
    call get_soap(n_sites_f, n_neigh, n_species_f, species, species_multiplicity, n_atom_pairs_f, mask, &
                  rjs, thetas, phis, alpha_max, l_max_f, rcut_hard, rcut_soft, nf, global_scaling, &
                  atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                  amplitude_scaling, radial_enhancement_f, central_weight, trim(basis), &
                  trim(scaling_mode), .false., .false., compress_soap_f, compression_nonzero, compress_i, compress_j, &
                  compress_el, soap, soap_cart_der)

    do i = 1, n_sites_f
      do j = 1, feature_count_f
        soap_out((i - 1) * feature_count_f + j) = soap(j, i)
      end do
    end do
  end subroutine soap_turbo_reference

end module mdescriptor_soap_turbo_reference
