/*
 * Copyright (c) 2010-2013 Centre National de la Recherche Scientifique.
 * written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
 * 
 * nathanael.schaeffer@ujf-grenoble.fr
 * 
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 * 
 */

/* shtns_numpy.i : SWIG interface to Python using NumPy */

/* TODO and known problems :
 * - alignement on 16 bytes of NumPy arrays is not guaranteed. It should however work on 64bit systems or on modern 32bit systems.
 * - you may have to adjust the path below to include the header file "arrayobject.h" from the NumPy package.
 */

%module (docstring="Python/NumPy interface to the SHTns spherical harmonic transform library") shtns

%init{
	import_array();		// required by NumPy
}

%pythoncode{
	import numpy as np
}

%{

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "sht_private.h"

// variables used for exception handling.
static int shtns_error = 0;
static char* shtns_err_msg;
static char msg_buffer[128];
static char msg_grid_err[] = "Grid not set. Call .set_grid() mehtod.";
static char msg_numpy_arr[] = "Numpy array expected.";
static char msg_rot_err[] = "truncation must be triangular (lmax=mmax, mres=1)";

static void throw_exception(int error, int iarg, char* msg)
{
	shtns_error = error;
	shtns_err_msg = msg;
	if (iarg > 0) {
		sprintf(msg_buffer, "arg #%d : %.100s", iarg, msg);
		shtns_err_msg = msg_buffer;
	}
}

static int check_spatial(int i, PyObject *a, int size) {
	if (size == 0) {
		throw_exception(SWIG_RuntimeError,0,msg_grid_err);	return 0;
	}
	if (!PyArray_Check(a)) {
		throw_exception(SWIG_TypeError,i,msg_numpy_arr);		return 0;
	}
	if (PyArray_TYPE((PyArrayObject *) a) != NPY_DOUBLE) {
		throw_exception(SWIG_TypeError,i,"spatial array must consist of float.");		return 0;
	}
	if (!PyArray_ISCONTIGUOUS((PyArrayObject *) a)) {
		throw_exception(SWIG_RuntimeError,i,"spatial array not contiguous. Use 'b=a.copy()' to copy a to a contiguous array b.");		return 0;
	}
	if (PyArray_SIZE((PyArrayObject *) a) != size) {
		throw_exception(SWIG_RuntimeError,i,"spatial array has wrong size");		return 0;
	}
	return 1;
}

static int check_spectral(int i, PyObject *a, int size) {
	if (!PyArray_Check(a)) {
		throw_exception(SWIG_RuntimeError,i,msg_numpy_arr);		return 0;
	}
	if (PyArray_TYPE((PyArrayObject *) a) != NPY_CDOUBLE) {
		throw_exception(SWIG_RuntimeError,i,"spectral array must consist of complex float. Create with: 'sh.spec_array()'");		return 0;
	}
	if (!PyArray_ISCONTIGUOUS((PyArrayObject *) a)) {
		throw_exception(SWIG_RuntimeError,i,"spactral array not contiguous. Use .copy() to copy to a contiguous array.");		return 0;
	}
	if (PyArray_SIZE((PyArrayObject *) a) != size) {
		throw_exception(SWIG_RuntimeError,i,"spectral array has wrong size");		return 0;
	}
	return 1;
}

inline static void* PyArray_Data(PyObject *a) {
	return PyArray_DATA((PyArrayObject*) a);
}

inline static PyObject* SpecArray_New(int size) {
	npy_intp dims = size;
	npy_intp strides = sizeof(cplx);
	return PyArray_New(&PyArray_Type, 1, &dims, NPY_CDOUBLE, &strides, NULL, strides, 0, NULL);	
}

inline static PyObject* SpatArray_New(int size) {
	npy_intp dims = size;
	npy_intp strides = sizeof(double);
	return PyArray_New(&PyArray_Type, 1, &dims, NPY_DOUBLE, &strides, NULL, strides, 0, NULL);	
}

%}

// main object is renamed to sht.
%rename("sht") shtns_info;
%ignore SHT_NATIVE_LAYOUT;
%ignore nlat_2;
%ignore lmidx;
%ignore li;
%ignore mi;
%ignore ct;
%ignore st;

%rename(print_version) shtns_print_version;
%rename(set_verbosity) shtns_verbose;

%feature("autodoc");
%include "shtns.h"
%include "exception.i"


%extend shtns_info {
	%exception {
		shtns_error = 0;	// clear exception
		$function
		if (shtns_error) {	// test for exception
			SWIG_exception(shtns_error, shtns_err_msg);		return NULL;
		}
	}

	%pythonappend shtns_info %{
		## array giving the degree of spherical harmonic coefficients.
		self.l = np.zeros(self.nlm, dtype=np.int32)
		## array giving the order of spherical harmonic coefficients.
		self.m = np.zeros(self.nlm, dtype=np.int32)
		for mloop in range(0, self.mmax*self.mres+1, self.mres):
			for lloop in range(mloop, self.lmax+1):
				ii = self.idx(lloop,mloop)
				self.m[ii] = mloop
				self.l[ii] = lloop
		self.m.flags.writeable = False		# prevent writing in m and l arrays
		self.l.flags.writeable = False
	%}
	%feature("kwargs") shtns_info;
	shtns_info(int lmax, int mmax=-1, int mres=1, int norm=sht_orthonormal, int nthreads=0) {	// default arguments : mmax, mres and norm
		if (lmax < 2) {
			throw_exception(SWIG_ValueError,1,"lmax < 2 not allowed");	return NULL;
		}
		if (mres <= 0) {
			throw_exception(SWIG_ValueError,3,"mres <= 0 invalid");	return NULL;
		}
		if (mmax < 0) mmax = lmax/mres;		// default mmax
		if (mmax*mres > lmax) {
			throw_exception(SWIG_ValueError,1,"lmax < mmax*mres invalid");	return NULL;
		}
		shtns_use_threads(nthreads);		// use nthreads openmp threads if available (0 means auto)
		return shtns_create(lmax, mmax, mres, norm);
	}

	~shtns_info() {
		shtns_destroy($self);		// free memory.
	}

	%pythonappend set_grid %{
		## array giving the cosine of the colatitude for the grid.
		self.cos_theta = self.__ct()
		self.cos_theta.flags.writeable = False
		## shape of a spatial array for the grid (tuple of 2 values).
		self.spat_shape = tuple(self.__spat_shape())
		if self.nphi == 1:		# override spatial shape when nphi==1
			self.spat_shape = (self.nlat, 1)
			if flags & SHT_THETA_CONTIGUOUS: self.spat_shape = (1, self.nlat)
	%}
	%apply int *OUTPUT { int *nlat_out };
	%apply int *OUTPUT { int *nphi_out };
	%feature("kwargs") set_grid;
	void set_grid(int nlat=0, int nphi=0, int flags=sht_quick_init, double polar_opt=1.0e-8, int nl_order=1, int *nlat_out, int *nphi_out) {	// default arguments
		if (nlat != 0) {
			if (nlat <= $self->lmax) {	// nlat too small
				throw_exception(SWIG_ValueError,1,"nlat <= lmax");		return;
			}
			if (nlat & 1) {		// nlat must be even
				throw_exception(SWIG_ValueError,1,"nlat must be even");		return;
			}
		}
		if ((nphi != 0) && (nphi <= $self->mmax *2)) {		// nphi too small
			throw_exception(SWIG_ValueError,2,"nphi <= 2*mmax");	return;
		}
		if (!(flags & SHT_THETA_CONTIGUOUS))  flags |= SHT_PHI_CONTIGUOUS;	// default to SHT_PHI_CONTIGUOUS.
		*nlat_out = nlat;		*nphi_out = nphi;
		shtns_set_grid_auto($self, flags, polar_opt, nl_order, nlat_out, nphi_out);
	}

	void print_info() {
		shtns_print_cfg($self);
	}
	double sh00_1() {
		return sh00_1($self);
	}
	double sh10_ct() {
		return sh10_ct($self);
	}
	double sh11_st() {
		return sh11_st($self);
	}
	double shlm_e1(unsigned l, unsigned m) {
		return shlm_e1($self, l, m);
	}

	/* returns useful data */
	PyObject* __ct() {		// grid must have been initialized.
		PyObject *obj;
		double *ct;
		int i;
		if ($self->nlat == 0) {	// no grid
			throw_exception(SWIG_RuntimeError,0,msg_grid_err);
			return NULL;
		}
		obj = SpatArray_New($self->nlat);
		ct = (double*) PyArray_Data(obj);
		for (i=0; i<$self->nlat; i++)		ct[i] = $self->ct[i];		// copy
		return obj;
	}
	PyObject* gauss_wts() {		// gauss grid must have been initialized.
		PyObject *obj;
		if ($self->nlat == 0) {	// no grid
			throw_exception(SWIG_RuntimeError,0,msg_grid_err);
			return NULL;
		}
		if ($self->wg == NULL) {
			throw_exception(SWIG_RuntimeError,0,"not a gauss grid");
			return NULL;
		}
		obj = SpatArray_New($self->nlat_2);
		shtns_gauss_wts($self, PyArray_Data(obj));
		return obj;
	}
	PyObject* mul_ct_matrix() {
		PyObject *mx = SpatArray_New(2*$self->nlm);
		mul_ct_matrix($self, PyArray_Data(mx));
		return mx;
	}
	PyObject* st_dt_matrix() {
		PyObject *mx = SpatArray_New(2*$self->nlm);
		st_dt_matrix($self, PyArray_Data(mx));
		return mx;
	}

	%apply int *OUTPUT { int *dim0 };
	%apply int *OUTPUT { int *dim1 };
	void __spat_shape(int *dim0, int *dim1) {
		*dim0 = $self->nphi;	*dim1 = $self->nlat;
		if ($self->fftc_mode == 1) {	// phi-contiguous
			*dim0 = $self->nlat;		*dim1 = $self->nphi;
		}
	}

	%pythoncode %{
		def spec_array(self, im=-1):
			"""return a numpy array of spherical harmonic coefficients (complex). Adress coefficients with index sh.idx(l,m)
			   if optional argument im is given, the spectral array is restricted to order im*mres."""
			if im<0:
				return np.zeros(self.nlm, dtype=complex)
			else:
				return np.zeros(self.lmax + 1 - im*self.mres, dtype=complex)
		
		def spat_array(self):
			"""return a numpy array of 2D spatial field."""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			return np.zeros(self.spat_shape)
	%}

	// returns the index in a spectral array of (l,m) coefficient.
	int idx(unsigned l, unsigned m) {
		if (l > $self->lmax) {
			throw_exception(SWIG_ValueError,1,"l invalid");	return 0;
		}
		if ( (m > l) || (m > $self->mmax * $self->mres) || (m % $self->mres != 0) ) {
			throw_exception(SWIG_ValueError,2,"m invalid");	return 0;
		}
		return LM($self, l, m);
	}

	/* scalar transforms */
	void spat_to_SH(PyObject *Vr, PyObject *Qlm) {
		if (check_spatial(1,Vr, $self->nspat) && check_spectral(2,Qlm, $self->nlm))
			spat_to_SH($self, PyArray_Data(Vr), PyArray_Data(Qlm));
	}
	void SH_to_spat(PyObject *Qlm, PyObject *Vr) {
		if (check_spatial(2,Vr, $self->nspat) && check_spectral(1,Qlm, $self->nlm))
			SH_to_spat($self, PyArray_Data(Qlm), PyArray_Data(Vr));
	}
	/* complex transforms */
	void spat_cplx_to_SH(PyObject *z, PyObject *alm) {
		int n = $self->lmax + 1;
		if (check_spectral(1,z, $self->nspat) && check_spectral(2,alm, n*n))
			spat_cplx_to_SH($self, PyArray_Data(z), PyArray_Data(alm));
	}
	void SH_to_spat_cplx(PyObject *alm, PyObject *z) {
		int n = $self->lmax + 1;
		if (check_spectral(2,z, $self->nspat) && check_spectral(1,alm, n*n))
			SH_to_spat_cplx($self, PyArray_Data(alm), PyArray_Data(z));
	}
	/*void SH_to_spat_grad(PyObject *alm, PyObject *gt, PyObject *gp) {
		if (check_spatial(3,gp, $self->nspat) && check_spatial(2,gt, $self->nspat) && check_spectral(1,alm, $self->nlm))
			SH_to_spat_grad($self, PyArray_Data(alm), PyArray_Data(gt), PyArray_Data(gp));
	}*/

	/* 2D vectors */
	void spat_to_SHsphtor(PyObject *Vt, PyObject *Vp, PyObject *Slm, PyObject *Tlm) {
		if (check_spatial(1,Vt, $self->nspat) && check_spatial(2,Vp, $self->nspat) && check_spectral(3,Slm, $self->nlm) && check_spectral(4,Tlm, $self->nlm))
			spat_to_SHsphtor($self, PyArray_Data(Vt), PyArray_Data(Vp), PyArray_Data(Slm), PyArray_Data(Tlm));
	}
	void SHsphtor_to_spat(PyObject *Slm, PyObject *Tlm, PyObject *Vt, PyObject *Vp) {
		if (check_spatial(3,Vt, $self->nspat) && check_spatial(4,Vp, $self->nspat) && check_spectral(1,Slm, $self->nlm) && check_spectral(2,Tlm, $self->nlm))
			SHsphtor_to_spat($self, PyArray_Data(Slm), PyArray_Data(Tlm), PyArray_Data(Vt), PyArray_Data(Vp));
	}
	void SHsph_to_spat(PyObject *Slm, PyObject *Vt, PyObject *Vp) {
		if (check_spatial(2,Vt, $self->nspat) && check_spatial(3,Vp, $self->nspat) && check_spectral(1,Slm, $self->nlm))
		SHsph_to_spat($self, PyArray_Data(Slm), PyArray_Data(Vt), PyArray_Data(Vp));
	}
	void SHtor_to_spat(PyObject *Tlm, PyObject *Vt, PyObject *Vp) {
		if (check_spatial(2,Vt, $self->nspat) && check_spatial(3,Vp, $self->nspat) && check_spectral(1,Tlm, $self->nlm))
		SHtor_to_spat($self, PyArray_Data(Tlm), PyArray_Data(Vt), PyArray_Data(Vp));
	}

	/* 3D vectors */
	void spat_to_SHqst(PyObject *Vr, PyObject *Vt, PyObject *Vp, PyObject *Qlm, PyObject *Slm, PyObject *Tlm) {
		if (check_spatial(1,Vr, $self->nspat) && check_spatial(2,Vt, $self->nspat) && check_spatial(3,Vp, $self->nspat)
			&& check_spectral(4,Qlm, $self->nlm) && check_spectral(5,Slm, $self->nlm) && check_spectral(6,Tlm, $self->nlm))
		spat_to_SHqst($self, PyArray_Data(Vr), PyArray_Data(Vt), PyArray_Data(Vp), PyArray_Data(Qlm), PyArray_Data(Slm), PyArray_Data(Tlm));
	}
	void SHqst_to_spat(PyObject *Qlm, PyObject *Slm, PyObject *Tlm, PyObject *Vr, PyObject *Vt, PyObject *Vp) {
		if (check_spatial(4,Vr, $self->nspat) && check_spatial(5,Vt, $self->nspat) && check_spatial(6,Vp, $self->nspat)
			&& check_spectral(1,Qlm, $self->nlm) && check_spectral(2,Slm, $self->nlm) && check_spectral(3,Tlm, $self->nlm))
		SHqst_to_spat($self, PyArray_Data(Qlm), PyArray_Data(Slm), PyArray_Data(Tlm), PyArray_Data(Vr), PyArray_Data(Vt), PyArray_Data(Vp));
	}

	%pythoncode %{
		def synth(self,*arg):
			"""
			spectral to spatial transform, for scalar or vector data.
			v = synth(qlm) : compute the spatial representation of the scalar qlm
			vtheta,vphi = synth(slm,tlm) : compute the 2D spatial vector from its spectral spheroidal/toroidal scalars (slm,tlm)
			vr,vtheta,vphi = synth(qlm,slm,tlm) : compute the 3D spatial vector from its spectral radial/spheroidal/toroidal scalars (qlm,slm,tlm)
			"""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			n = len(arg)
			if (n>3) or (n<1): raise RuntimeError("1,2 or 3 arguments required.")
			q = list(arg)
			for i in range(0,n):
				if q[i].size != self.nlm: raise RuntimeError("spectral array has wrong size.")
				if q[i].dtype.num != np.dtype('complex128').num: raise RuntimeError("spectral array should be dtype=complex.")
				if q[i].flags.contiguous == False: q[i] = q[i].copy()		# contiguous array required.
			if n==1:	#scalar transform
				vr = np.empty(self.spat_shape)
				self.SH_to_spat(q[0],vr)
				return vr
			elif n==2:	# 2D vector transform
				vt = np.empty(self.spat_shape)		# v_theta
				vp = np.empty(self.spat_shape)		# v_phi
				self.SHsphtor_to_spat(q[0],q[1],vt,vp)
				return vt,vp
			else:		# 3D vector transform
				vr = np.empty(self.spat_shape)		# v_r
				vt = np.empty(self.spat_shape)		# v_theta
				vp = np.empty(self.spat_shape)		# v_phi
				self.SHqst_to_spat(q[0],q[1],q[2],vr,vt,vp)
				return vr,vt,vp

		def analys(self,*arg):
			"""
			spatial to spectral transform, for scalar or vector data.
			qlm = analys(q) : compute the spherical harmonic representation of the scalar q
			slm,tlm = analys(vtheta,vphi) : compute the spectral spheroidal/toroidal scalars (slm,tlm) from 2D vector components (vtheta, vphi)
			qlm,slm,tlm = synth(vr,vtheta,vphi) : compute the spectral radial/spheroidal/toroidal scalars (qlm,slm,tlm) from 3D vector components (vr,vtheta,vphi)
			"""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			if abs(self.cos_theta[0]) == 1: raise RuntimeError("Analysis not allowed with sht_reg_poles grid.")
			n = len(arg)
			if (n>3) or (n<1): raise RuntimeError("1,2 or 3 arguments required.")
			v = list(arg)
			for i in range(0,n):
				if v[i].shape != self.spat_shape: raise RuntimeError("spatial array has wrong shape.")
				if v[i].dtype.num != np.dtype('float64').num: raise RuntimeError("spatial array should be dtype=float64.")
				if v[i].flags.contiguous == False: v[i] = v[i].copy()		# contiguous array required.
			if n==1:
				q = np.empty(self.nlm, dtype=complex)
				self.spat_to_SH(v[0],q)
				return q
			elif n==2:
				s = np.empty(self.nlm, dtype=complex)
				t = np.empty(self.nlm, dtype=complex)
				self.spat_to_SHsphtor(v[0],v[1],s,t)
				return s,t
			else:
				q = np.empty(self.nlm, dtype=complex)
				s = np.empty(self.nlm, dtype=complex)
				t = np.empty(self.nlm, dtype=complex)
				self.spat_to_SHqst(v[0],v[1],v[2],q,s,t)
				return q,s,t

		def synth_grad(self,slm):
			"""(vtheta,vphi) = synth_grad(sht self, slm) : compute the spatial representation of the gradient of slm"""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			if slm.size != self.nlm: raise RuntimeError("spectral array has wrong size.")
			if slm.dtype.num != np.dtype('complex128').num: raise RuntimeError("spectral array should be dtype=complex.")
			if slm.flags.contiguous == False: slm = slm.copy()		# contiguous array required.
			vt = np.empty(self.spat_shape)
			vp = np.empty(self.spat_shape)
			self.SHsph_to_spat(slm,vt,vp)
			return vt,vp

		def synth_cplx(self,alm):
			"""
			spectral to spatial transform, for complex valued scalar data.
			z = synth(alm) : compute the spatial representation of the scalar alm
			"""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			if self.lmax != self.mmax: raise RuntimeError("complex SH requires lmax=mmax and mres=1.")
			if alm.size != (self.lmax+1)**2: raise RuntimeError("spectral array has wrong size.")
			if alm.dtype.num != np.dtype('complex128').num: raise RuntimeError("spectral array should be dtype=complex.")
			if alm.flags.contiguous == False: alm = alm.copy()		# contiguous array required.
			z = np.empty(self.spat_shape, dtype=complex)
			self.SH_to_spat_cplx(alm,z)
			return z

		def analys_cplx(self,z):
			"""
			spatial to spectral transform, for complex valued scalar data.
			alm = analys(z) : compute the spherical harmonic representation of the complex scalar z
			"""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			if self.lmax != self.mmax: raise RuntimeError("complex SH requires lmax=mmax and mres=1.")
			if z.shape != self.spat_shape: raise RuntimeError("spatial array has wrong shape.")
			if z.dtype.num != np.dtype('complex128').num: raise RuntimeError("spatial array should be dtype=complex128.")
			if z.flags.contiguous == False: z = z.copy()		# contiguous array required.
			alm = np.empty((self.lmax+1)**2, dtype=complex)
			self.spat_cplx_to_SH(z,alm)
			return alm

		def zidx(self, l,m):
			"""
			zidx(sht self, int l, int m) -> int : compute the index l*(l+1)+m in a complex spherical harmonic expansion
			"""
			l = np.asarray(l)
			m = np.asarray(m)
			if (l>self.lmax).any() or (abs(m)>l).any() : raise RuntimeError("invalid range for l,m")
			return l*(l+1)+m

		def zlm(self, idx):
			"""
			zlm(sht self, int idx) -> (int,int) : returns the l and m corresponding to the given index in complex spherical harmonic expansion
			"""
			idx = np.asarray(idx)
			if (idx >= (self.lmax+1)**2).any() or (idx < 0).any() : raise RuntimeError("invalid range for l,m")
			l = np.sqrt(idx).astype(int)
			m = idx - l*(l+1)
			return l,m

		def spec_array_cplx(self):
			"""return a numpy array that can hold the spectral representation of a complex scalar spatial field."""
			return np.zeros((self.lmax+1)**2, dtype=complex)
		
		def spat_array_cplx(self):
			"""return a numpy array of 2D complex spatial field."""
			if self.nlat == 0: raise RuntimeError("Grid not set. Call .set_grid() mehtod.")
			return np.zeros(self.spat_shape, dtype=complex128)
	%}

	/* local evaluations */
	double SH_to_point(PyObject *Qlm, double cost, double phi) {
		if (check_spectral(1,Qlm, $self->nlm))	return SH_to_point($self, PyArray_Data(Qlm), cost, phi);
		return 0.0;
	}
	%apply double *OUTPUT { double *vr };
	%apply double *OUTPUT { double *vt };
	%apply double *OUTPUT { double *vp };
	void SH_to_grad_point(PyObject *DrSlm, PyObject *Slm, double cost, double phi, double *vr, double *vt, double *vp) {
		if (check_spectral(1,DrSlm, $self->nlm) && check_spectral(2,Slm, $self->nlm))
			SH_to_grad_point($self, PyArray_Data(DrSlm), PyArray_Data(Slm), cost, phi, vr, vt, vp);
	}
	void SHqst_to_point(PyObject *Qlm, PyObject *Slm, PyObject *Tlm,
					double cost, double phi, double *vr, double *vt, double *vp) {
		if (check_spectral(1,Qlm, $self->nlm) && check_spectral(2,Slm, $self->nlm) && check_spectral(3,Tlm, $self->nlm))
			SHqst_to_point($self, PyArray_Data(Qlm), PyArray_Data(Slm), PyArray_Data(Tlm), cost, phi, vr, vt, vp);
	}
	%clear double *vr;
	%clear double *vt;
	%clear double *vp;
	
	/* _to_lat */
	void SH_to_lat(PyObject *Qlm, double cost, PyObject *Vr) {
		if (check_spatial(3,Vr, PyArray_SIZE((PyArrayObject *) Vr)) && check_spectral(1,Qlm, $self->nlm))
			SH_to_lat($self, PyArray_Data(Qlm), cost, PyArray_Data(Vr), PyArray_SIZE((PyArrayObject *) Vr), $self->lmax, $self->mmax);
	}
	void SHqst_to_lat(PyObject *Qlm, PyObject *Slm, PyObject *Tlm, double cost, PyObject *Vr, PyObject *Vt, PyObject *Vp) {
		int nphi = PyArray_SIZE((PyArrayObject *) Vr);
		if (check_spatial(5,Vr, nphi) && check_spatial(6,Vt, nphi) && check_spatial(7,Vp, nphi)
			&& check_spectral(1,Qlm, $self->nlm) && check_spectral(2,Slm, $self->nlm) && check_spectral(3,Tlm, $self->nlm))
			SHqst_to_lat($self, PyArray_Data(Qlm), PyArray_Data(Slm), PyArray_Data(Tlm), cost, PyArray_Data(Vr), PyArray_Data(Vt), PyArray_Data(Vp), nphi, $self->lmax, $self->mmax);
	}

	/* rotation of SH representations (experimental) */
	PyObject* Zrotate(PyObject *Qlm, double alpha) {
		if (check_spectral(1,Qlm, $self->nlm)) {
			PyObject *Rlm = SpecArray_New($self->nlm);
			SH_Zrotate($self, PyArray_Data(Qlm), alpha, PyArray_Data(Rlm));
			return Rlm;
		}
		return NULL;
	}
	PyObject* Yrotate(PyObject *Qlm, double alpha) {
		if (($self->mres != 1)||($self->mmax != $self->lmax)) {
			throw_exception(SWIG_RuntimeError,0,msg_rot_err);	return NULL;
		}
		if (check_spectral(1,Qlm, $self->nlm)) {
			PyObject *Rlm = SpecArray_New($self->nlm);
			SH_Yrotate($self, PyArray_Data(Qlm), alpha, PyArray_Data(Rlm));
			return Rlm;
		}
		return NULL;
	}
	PyObject* Yrotate90(PyObject *Qlm) {
		if (($self->mres != 1)||($self->mmax != $self->lmax)) {
			throw_exception(SWIG_RuntimeError,0,msg_rot_err);	return NULL;
		}
		if (check_spectral(1,Qlm, $self->nlm)) {
			PyObject *Rlm = SpecArray_New($self->nlm);
			SH_Yrotate90($self, PyArray_Data(Qlm), PyArray_Data(Rlm));
			return Rlm;
		}
		return NULL;
	}
	PyObject* Xrotate90(PyObject *Qlm) {
		if (($self->mres != 1)||($self->mmax != $self->lmax)) {
			throw_exception(SWIG_RuntimeError,0,msg_rot_err);	return NULL;
		}
		if (check_spectral(1,Qlm, $self->nlm)) {
			PyObject *Rlm = SpecArray_New($self->nlm);
			SH_Xrotate90($self, PyArray_Data(Qlm), PyArray_Data(Rlm));
			return Rlm;
		}
		return NULL;
	}

	/* multiplication by l+1 l-1 matrix (mul_ct_matrix or st_dt_matrix) */
	PyObject* SH_mul_mx(PyObject *mx, PyObject *Qlm) {
		if (check_spectral(2,Qlm, $self->nlm) && check_spatial(1, mx, 2* $self->nlm)) {
			PyObject *Rlm = SpecArray_New($self->nlm);
			SH_mul_mx($self, PyArray_Data(mx), PyArray_Data(Qlm), PyArray_Data(Rlm));
			return Rlm;
		}
		return NULL;
	}

	/* Legendre transforms (no fft) at given order m */
	void spat_to_SH_m(PyObject *Vr, PyObject *Qlm, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;
		if ((im_ >= 0) && check_spectral(1,Vr, $self->nlat) && check_spectral(2,Qlm, ltr+1 - im_*$self->mres))
			spat_to_SH_ml($self, im_, PyArray_Data(Vr), PyArray_Data(Qlm), ltr);
	}
	void SH_to_spat_m(PyObject *Qlm, PyObject *Vr, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;
		if ((im_ >= 0) && check_spectral(2,Vr, $self->nlat) && check_spectral(1,Qlm, ltr+1 - im_*$self->mres))
			SH_to_spat_ml($self, im_, PyArray_Data(Qlm), PyArray_Data(Vr), ltr);
	}
	void spat_to_SHsphtor_m(PyObject *Vt, PyObject *Vp, PyObject *Slm, PyObject *Tlm, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;		int nelem = ltr+1 - im_*$self->mres;
		if ((im_ >= 0) && check_spectral(1,Vt, $self->nlat) && check_spectral(2,Vp, $self->nlat) && check_spectral(3,Slm, nelem) && check_spectral(4,Tlm, nelem))
			spat_to_SHsphtor_ml($self, im_, PyArray_Data(Vt), PyArray_Data(Vp), PyArray_Data(Slm), PyArray_Data(Tlm), ltr);
	}
	void SHsphtor_to_spat_m(PyObject *Slm, PyObject *Tlm, PyObject *Vt, PyObject *Vp, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;		int nelem = ltr+1 - im_*$self->mres;
		if ((im_ >= 0) && check_spectral(3,Vt, $self->nlat) && check_spectral(4,Vp, $self->nlat) && check_spectral(1,Slm, nelem) && check_spectral(2,Tlm, nelem))
			SHsphtor_to_spat_ml($self, im_, PyArray_Data(Slm), PyArray_Data(Tlm), PyArray_Data(Vt), PyArray_Data(Vp), ltr);
	}
	void SHsph_to_spat_m(PyObject *Slm, PyObject *Vt, PyObject *Vp, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;		int nelem = ltr+1 - im_*$self->mres;
		if ((im_ >= 0) && check_spectral(2,Vt, $self->nlat) && check_spectral(3,Vp, $self->nlat) && check_spectral(1,Slm, nelem))
		SHsph_to_spat_ml($self, im_, PyArray_Data(Slm), PyArray_Data(Vt), PyArray_Data(Vp), ltr);
	}
	void SHtor_to_spat_m(PyObject *Tlm, PyObject *Vt, PyObject *Vp, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;		int nelem = ltr+1 - im_*$self->mres;
		if ((im_ >= 0) && check_spectral(2,Vt, $self->nlat) && check_spectral(3,Vp, $self->nlat) && check_spectral(1,Tlm, nelem))
		SHtor_to_spat_ml($self, im_, PyArray_Data(Tlm), PyArray_Data(Vt), PyArray_Data(Vp), ltr);
	}
	void spat_to_SHqst_m(PyObject *Vr, PyObject *Vt, PyObject *Vp, PyObject *Qlm, PyObject *Slm, PyObject *Tlm, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;		int nelem = ltr+1 - im_*$self->mres;
		if ((im_ >= 0) && check_spectral(1,Vr, $self->nlat) && check_spectral(2,Vt, $self->nlat) && check_spectral(3,Vp, $self->nlat)
			&& check_spectral(4,Qlm, nelem) && check_spectral(5,Slm, nelem) && check_spectral(6,Tlm, nelem))
		spat_to_SHqst_ml($self, im_, PyArray_Data(Vr), PyArray_Data(Vt), PyArray_Data(Vp), PyArray_Data(Qlm), PyArray_Data(Slm), PyArray_Data(Tlm), ltr);
	}
	void SHqst_to_spat_m(PyObject *Qlm, PyObject *Slm, PyObject *Tlm, PyObject *Vr, PyObject *Vt, PyObject *Vp, PyObject *im) {
		int im_ = PyLong_AsLong(im);		int ltr = $self->lmax;		int nelem = ltr+1 - im_*$self->mres;
		if ((im_ >= 0) && check_spectral(4,Vr, $self->nlat) && check_spectral(5,Vt, $self->nlat) && check_spectral(6,Vp, $self->nlat)
			&& check_spectral(1,Qlm, nelem) && check_spectral(2,Slm, nelem) && check_spectral(3,Tlm, nelem))
		SHqst_to_spat_ml($self, im_, PyArray_Data(Qlm), PyArray_Data(Slm), PyArray_Data(Tlm), PyArray_Data(Vr), PyArray_Data(Vt), PyArray_Data(Vp), ltr);
	}

};
