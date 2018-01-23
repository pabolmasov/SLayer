/*
 * Copyright (c) 2010-2016 Centre National de la Recherche Scientifique.
 * written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
 * 
 * nathanael.schaeffer@univ-grenoble-alpes.fr
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

# This file is meta-code for SHT.c (spherical harmonic transform).
# it is intended for "make" to generate C code for 3 similar SHT functions,
# (namely spat_to_SH [Q tag]), spat_to_SHsphtor [V tag], spat_to_SH3 [both Q&V tags])
# from one generic function + tags.
# Basically, there are tags at the beginning of lines (Q,V) that are information
# to keep or remove the line depending on the function to build. (Q for scalar, V for vector, # for comment)
#
//////////////////////////////////////////////////

QX	static void GEN3(spat_to_SH_fly,NWAY,SUFFIX)(shtns_cfg shtns, double *Vr, cplx *Qlm, const long int llim) {
VX	static void GEN3(spat_to_SHsphtor_fly,NWAY,SUFFIX)(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, const long int llim) {
3	static void GEN3(spat_to_SHqst_fly,NWAY,SUFFIX)(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, const long int llim) {

Q	double *BrF;		// contains the Fourier transformed data
V	double *BtF, *BpF;	// contains the Fourier transformed data
	double *alm, *al;
	double *wg, *ct, *st;
V	double *l_2;
	long int nk, k, l,m;
  #ifndef SHT_AXISYM
	unsigned imlim, im;
	int k_inc, m_inc;
  #else
	const int k_inc = 1;
  #endif
Q	rnd qq[2*llim+4];
V	rnd vw[4*llim+8];

Q	double rer[NLAT_2 + NWAY*VSIZE2] SSE;
Q	double ror[NLAT_2 + NWAY*VSIZE2] SSE;
V	double ter[NLAT_2 + NWAY*VSIZE2] SSE;
V	double tor[NLAT_2 + NWAY*VSIZE2] SSE;
V	double per[NLAT_2 + NWAY*VSIZE2] SSE;
V	double por[NLAT_2 + NWAY*VSIZE2] SSE;
  #ifndef SHT_AXISYM
Q	double rei[NLAT_2 + NWAY*VSIZE2] SSE;
Q	double roi[NLAT_2 + NWAY*VSIZE2] SSE;
V	double tei[NLAT_2 + NWAY*VSIZE2] SSE;
V	double toi[NLAT_2 + NWAY*VSIZE2] SSE;
V	double pei[NLAT_2 + NWAY*VSIZE2] SSE;
V	double poi[NLAT_2 + NWAY*VSIZE2] SSE;
  #endif

Q	BrF = Vr;
V	BtF = Vt;	BpF = Vp;
  #ifndef SHT_AXISYM
	if (shtns->fftc_mode >= 0) {
	    if (shtns->fftc_mode == 0) {	// in-place
Q			fftw_execute_dft(shtns->fftc,(cplx*)BrF, (cplx*)BrF);
V			fftw_execute_dft(shtns->fftc,(cplx*)BtF, (cplx*)BtF);
V			fftw_execute_dft(shtns->fftc,(cplx*)BpF, (cplx*)BpF);
		} else {	// alloc memory for the transpose FFT
			unsigned long nv = shtns->nspat;
QX			BrF = (double*) VMALLOC( nv * sizeof(double) );
VX			BtF = (double*) VMALLOC( 2*nv * sizeof(double) );
VX			BpF = BtF + nv;
3			BrF = (double*) VMALLOC( 3*nv * sizeof(double) );
3			BtF = BrF + nv;		BpF = BtF + nv;
Q			fftw_execute_split_dft(shtns->fftc, Vr+NPHI, Vr, ((double*)BrF)+1, ((double*)BrF));
V			fftw_execute_split_dft(shtns->fftc, Vt+NPHI, Vt, ((double*)BtF)+1, ((double*)BtF));
V			fftw_execute_split_dft(shtns->fftc, Vp+NPHI, Vp, ((double*)BpF)+1, ((double*)BpF));
	    }
	}
	imlim = MTR;
	#ifdef SHT_VAR_LTR
		if (imlim*MRES > (unsigned) llim) imlim = ((unsigned) llim)/MRES;		// 32bit mul and div should be faster
	#endif

	// ACCESS PATTERN
	k_inc = shtns->k_stride_a;
	m_inc = shtns->m_stride_a;
  #endif

	nk = NLAT_2;	// copy NLAT_2 to a local variable for faster access (inner loop limit)
	wg = shtns->wg;		ct = shtns->ct;		st = shtns->st;
	#if _GCC_VEC_
	  nk = ((unsigned) nk+(VSIZE2-1))/VSIZE2;
	#endif
V	l_2 = shtns->l_2;
		alm = shtns->blm;
		// compute symmetric and antisymmetric parts. (do not weight here, it is cheaper to weight y0)
V		SYM_ASYM_M0_V(BtF, ter, tor)
V		SYM_ASYM_M0_V(BpF, per, por)
Q		double r0 = 0.0;
Q		SYM_ASYM_M0_Q(BrF, rer, ror, r0)
Q		Qlm[0] = r0 * alm[0];			// l=0 is done.
		for (k=nk*VSIZE2; k<(nk-1+NWAY)*VSIZE2; ++k) {
Q			rer[k] = 0.0;		ror[k] = 0.0;
V			ter[k] = 0.0;		tor[k] = 0.0;
V			per[k] = 0.0;		por[k] = 0.0;
		}
Q		Qlm[0] = r0 * alm[0];				// l=0 is done.
V		Slm[0] = 0.0;		Tlm[0] = 0.0;		// l=0 is zero for the vector transform.
		k = 0;
		for (l=0;l<llim;++l) {
Q			qq[l] = vall(0.0);
V			vw[2*l] = vall(0.0);		vw[2*l+1] = vall(0.0);
		}
		do {
			al = alm;
			rnd cost[NWAY], y0[NWAY], y1[NWAY];
V			rnd sint[NWAY], dy0[NWAY], dy1[NWAY];
Q			rnd rerk[NWAY], rork[NWAY];		// help the compiler to cache into registers.
V			rnd terk[NWAY], tork[NWAY], perk[NWAY], pork[NWAY];
			for (int j=0; j<NWAY; ++j) {
				cost[j] = vread(ct, k+j);
				y0[j] = vall(al[0]) * vread(wg, k+j);		// weight of Gauss quadrature appears here
V				dy0[j] = vall(0.0);
V				sint[j] = -vread(st, k+j);
				y1[j] =  (vall(al[1])*y0[j]) * cost[j];
V				dy1[j] = (vall(al[1])*y0[j]) * sint[j];
Q				rerk[j] = vread(rer, k+j);		rork[j] = vread(ror, k+j);		// cache into registers.
V				terk[j] = vread(ter, k+j);		tork[j] = vread(tor, k+j);
V				perk[j] = vread(per, k+j);		pork[j] = vread(por, k+j);
			}
			al+=2;	l=1;
			while(l<llim) {
				for (int j=0; j<NWAY; ++j) {
V					dy0[j] = vall(al[1])*(cost[j]*dy1[j] + y1[j]*sint[j]) + vall(al[0])*dy0[j];
					y0[j]  = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					qq[l-1]   += y1[j]  * rork[j];
V					vw[2*l-2] += dy1[j] * terk[j];
V					vw[2*l-1] -= dy1[j] * perk[j];
				}
				for (int j=0; j<NWAY; ++j) {
V					dy1[j] = vall(al[3])*(cost[j]*dy0[j] + y0[j]*sint[j]) + vall(al[2])*dy1[j];
					y1[j]  = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					qq[l]     += y0[j]  * rerk[j];
V					vw[2*l]   += dy0[j] * tork[j];
V					vw[2*l+1] -= dy0[j] * pork[j];
				}
				al+=4;	l+=2;
			}
			if (l==llim) {
				for (int j=0; j<NWAY; ++j) {
Q					qq[l-1]   += y1[j]  * rork[j];
V					vw[2*l-2] += dy1[j] * terk[j];
V					vw[2*l-1] -= dy1[j] * perk[j];
				}
			}
			k+=NWAY;
		} while (k < nk);		// limit: k=nk-1   =>  k=nk-1+NWAY is never read.
		for (l=1; l<=llim; ++l) {
			#if _GCC_VEC_
Q				((v2d*)Qlm)[l] = v2d_reduce(qq[l-1], vall(0));
V				((v2d*)Slm)[l] = v2d_reduce(vw[2*l-2], vall(0)) * vdup(l_2[l]);
V				((v2d*)Tlm)[l] = v2d_reduce(vw[2*l-1], vall(0)) * vdup(l_2[l]);
			#else
Q				Qlm[l] = qq[l-1];
V				Slm[l] = vw[2*l-2]*l_2[l];		Tlm[l] = vw[2*l-1]*l_2[l];
			#endif
		}
		#ifdef SHT_VAR_LTR
			for (l=llim+1; l<= LMAX; ++l) {
Q				Qlm[l] = 0.0;
V				Slm[l] = 0.0;		Tlm[l] = 0.0;
			}
		#endif

  #ifndef SHT_AXISYM
	for (k=nk*VSIZE2; k<(nk-1+NWAY)*VSIZE2; ++k) {		// never written, so this is now done for all m's (real parts already zero)
Q		rei[k] = 0.0;		roi[k] = 0.0;
V		tei[k] = 0.0;		toi[k] = 0.0;
V		pei[k] = 0.0;		poi[k] = 0.0;
	}
	for (im=1;im<=imlim;++im) {
		m = im*MRES;
		l = shtns->tm[im] / VSIZE2;
		//alm = shtns->blm[im];
		alm += 2*(LMAX+1-m+MRES);
		// compute symmetric and anti-symmetric parts:
QX		SYM_ASYM_Q(BrF, rer, ror, rei, roi, l)
3		SYM_ASYM_Q3(BrF, rer, ror, rei, roi, l)
V		SYM_ASYM_V(BtF, ter, tor, tei, toi, l)
V		SYM_ASYM_V(BpF, per, por, pei, poi, l)
		k=l;
		#if _GCC_VEC_
Q			rnd* q = qq;
V			rnd* v = vw;
		#else
			l = LiM(shtns, m, im);
Q			double* q = (double *) qq; //&Qlm[LiM(shtns, m, im)];
V			double* v = (double *) vw;
		#endif
		for (l=llim+1-m; l>=0; l--) {
Q			q[0] = vall(0.0);		q[1] = vall(0.0);		q+=2;
V			v[0] = vall(0.0);		v[1] = vall(0.0);
V			v[2] = vall(0.0);		v[3] = vall(0.0);		v+=4;
		}
		do {
		#if _GCC_VEC_
Q			rnd* q = qq;
V			rnd* v = vw;
		#else
Q			double* q = (double *) qq; //&Qlm[LiM(shtns, m, im)];
V			double* v = (double *) vw;
		#endif
			al = alm;
			rnd cost[NWAY], y0[NWAY], y1[NWAY];
Q			rnd rerk[NWAY], reik[NWAY], rork[NWAY], roik[NWAY];		// help the compiler to cache into registers.
V			rnd terk[NWAY], teik[NWAY], tork[NWAY], toik[NWAY];
V			rnd perk[NWAY], peik[NWAY], pork[NWAY], poik[NWAY];
			for (int j=0; j<NWAY; ++j) {
				cost[j] = vread(st, k+j);
				y0[j] = vall(0.5);
			}
Q			l=m;
V			l=m-1;
			long int ny = 0;	// exponent to extend double precision range.
		if ((int)llim <= SHT_L_RESCALE_FLY) {
			do {		// sin(theta)^m
				if (l&1) for (int j=0; j<NWAY; ++j) y0[j] *= cost[j];
				for (int j=0; j<NWAY; ++j) cost[j] *= cost[j];
			} while(l >>= 1);
		} else {
			long int nsint = 0;
			do {		// sin(theta)^m		(use rescaling to avoid underflow)
				if (l&1) {
					for (int j=NWAY-1; j>=0; --j) y0[j] *= cost[j];
					ny += nsint;
					if (vlo(y0[NWAY-1]) < (SHT_ACCURACY+1.0/SHT_SCALE_FACTOR)) {
						ny--;
						for (int j=NWAY-1; j>=0; --j) y0[j] *= vall(SHT_SCALE_FACTOR);
					}
				}
				for (int j=NWAY-1; j>=0; --j) cost[j] *= cost[j];
				nsint += nsint;
				if (vlo(cost[NWAY-1]) < 1.0/SHT_SCALE_FACTOR) {
					nsint--;
					for (int j=NWAY-1; j>=0; --j) cost[j] *= vall(SHT_SCALE_FACTOR);
				}
			} while(l >>= 1);
		}
			for (int j=0; j<NWAY; ++j) {
				y0[j] *= vall(al[0]);
				cost[j] = vread(ct, k+j);
				y1[j]  = (vall(al[1])*y0[j]) *cost[j];
			}
			l=m;	al+=2;
			while ((ny<0) && (l<llim)) {		// ylm treated as zero and ignored if ny < 0
				for (int j=0; j<NWAY; ++j) {
					y0[j] = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
					y1[j] = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				l+=2;	al+=4;
				if (fabs(vlo(y0[NWAY-1])) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0) {		// rescale when value is significant
					++ny;
					for (int j=0; j<NWAY; ++j) {
						y0[j] *= vall(1.0/SHT_SCALE_FACTOR);		y1[j] *= vall(1.0/SHT_SCALE_FACTOR);
					}
				}
			}
		  if (ny == 0) {
Q			q+=2*(l-m);
V			v+=4*(l-m);
			for (int j=0; j<NWAY; ++j) {	// prefetch
				y0[j] *= vread(wg, k+j);		y1[j] *= vread(wg, k+j);		// weight appears here (must be after the previous accuracy loop).
Q				rerk[j] = vread( rer, k+j);		reik[j] = vread( rei, k+j);		rork[j] = vread( ror, k+j);		roik[j] = vread( roi, k+j);
V				terk[j] = vread( ter, k+j);		teik[j] = vread( tei, k+j);		tork[j] = vread( tor, k+j);		toik[j] = vread( toi, k+j);
V				perk[j] = vread( per, k+j);		peik[j] = vread( pei, k+j);		pork[j] = vread( por, k+j);		poik[j] = vread( poi, k+j);
			}
			while (l<llim) {	// compute even and odd parts
Q				for (int j=0; j<NWAY; ++j)	{	q[0] += y0[j] * rerk[j];	q[1] += y0[j] * reik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[0] += y0[j] * terk[j];	v[1] += y0[j] * teik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[2] += y0[j] * perk[j];	v[3] += y0[j] * peik[j];	}
				for (int j=0; j<NWAY; ++j) {
					y0[j] = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
Q				for (int j=0; j<NWAY; ++j)	{	q[2] += y1[j] * rork[j];	q[3] += y1[j] * roik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[4] += y1[j] * tork[j];	v[5] += y1[j] * toik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[6] += y1[j] * pork[j];	v[7] += y1[j] * poik[j];	}
Q				q+=4;
V				v+=8;
				for (int j=0; j<NWAY; ++j) {
					y1[j] = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				l+=2;	al+=4;
			}
V				for (int j=0; j<NWAY; ++j)	{	v[0] += y0[j] * terk[j];	v[1] += y0[j] * teik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[2] += y0[j] * perk[j];	v[3] += y0[j] * peik[j];	}
			if (l==llim) {
Q				for (int j=0; j<NWAY; ++j)	{	q[0] += y0[j] * rerk[j];	q[1] += y0[j] * reik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[4] += y1[j] * tork[j];	v[5] += y1[j] * toik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[6] += y1[j] * pork[j];	v[7] += y1[j] * poik[j];	}
			}
		  }
			k+=NWAY;
		} while (k < nk);		// limit: k=nk-1   =>  k=nk-1+NWAY is never read.
		l = LiM(shtns, m, im);
Q		v2d * const Ql = (v2d*) &Qlm[l];
V		v2d * const Sl = (v2d*) &Slm[l];
V		v2d * const Tl = (v2d*) &Tlm[l];
Q		#if _GCC_VEC_
Q			for (l=0; l<=llim-m; ++l) {
Q				Ql[l] = v2d_reduce(qq[2*l], qq[2*l+1]);
Q			}
Q		#else
Q			for (l=0; l<=llim-m; ++l) {
Q				Ql[l] = qq[2*l] + I*qq[2*l+1];
Q			}
Q		#endif

V		{	// convert from the two scalar SH to vector SH
V			// Slm = - (I*m*Wlm + MX*Vlm) / (l*(l+1))		=> why does this work ??? (aliasing of 1/sin(theta) ???)
V			// Tlm = - (I*m*Vlm - MX*Wlm) / (l*(l+1))
V			double* mx = shtns->mx_van + 2*LM(shtns,m,m);	//(im*(2*(LMAX+1)-(m+MRES))) + 2*m;
V			s2d em = vdup(m);
V			v2d vl = v2d_reduce(vw[0], vw[1]);
V			v2d wl = v2d_reduce(vw[2], vw[3]);
V			v2d sl = vdup( 0.0 );
V			v2d tl = vdup( 0.0 );
V			for (int l=0; l<=llim-m; l++) {
V				s2d mxu = vdup( mx[2*l] );
V				s2d mxl = vdup( mx[2*l+1] );		// mxl for next iteration
V				sl = addi( sl ,  em*wl );
V				tl = addi( tl ,  em*vl );
V				v2d sl1 =  mxl*vl;			// vs for next iter
V				v2d tl1 = -mxl*wl;			// wt for next iter
V				vl = v2d_reduce(vw[4*l+4], vw[4*l+5]);		// kept for next iteration
V				wl = v2d_reduce(vw[4*l+6], vw[4*l+7]);
V				sl += mxu*vl;
V				tl -= mxu*wl;
V				Sl[l] = -sl * vdup(l_2[l+m]);
V				Tl[l] = -tl * vdup(l_2[l+m]);
V				sl = sl1;
V				tl = tl1;
V			}
V		}

		#ifdef SHT_VAR_LTR
			for (l=llim+1-m; l<=LMAX-m; ++l) {
Q				Ql[l] = vdup(0.0);
V				Sl[l] = vdup(0.0);		Tl[l] = vdup(0.0);
			}
		#endif
	}
	#ifdef SHT_VAR_LTR
	if (imlim < MMAX) {
		im = imlim+1;
		l = LiM(shtns, im*MRES, im);
		do {
Q			((v2d*)Qlm)[l] = vdup(0.0);
V			((v2d*)Slm)[l] = vdup(0.0);		((v2d*)Tlm)[l] = vdup(0.0);
		} while(++l < shtns->nlm);
	}
	#endif

  	if (shtns->fftc_mode > 0) {		// free memory
Q	    VFREE(BrF);
VX	    VFREE(BtF);	// this frees also BpF.
	}
  #endif

  }


  #ifndef SHT_AXISYM

QX	static void GEN3(spat_to_SH_m_fly,NWAY,SUFFIX)(shtns_cfg shtns, int im, cplx *Vr, cplx *Qlm, long int llim) {
VX	static void GEN3(spat_to_SHsphtor_m_fly,NWAY,SUFFIX)(shtns_cfg shtns, int im, cplx *Vt, cplx *Vp, cplx *Slm, cplx *Tlm, long int llim) {
3	static void GEN3(spat_to_SHqst_m_fly,NWAY,SUFFIX)(shtns_cfg shtns, int im, cplx *Vr, cplx *Vt, cplx *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, long int llim) {

	double *alm, *al;
	double *wg, *ct, *st;
V	double *l_2;
	long int nk, k, l,m;
	double alm0_rescale;
Q	rnd qq[2*llim+4];
V	rnd vw[4*llim+8];

Q	double rer[NLAT_2 + NWAY*VSIZE2] SSE;
Q	double ror[NLAT_2 + NWAY*VSIZE2] SSE;
V	double ter[NLAT_2 + NWAY*VSIZE2] SSE;
V	double tor[NLAT_2 + NWAY*VSIZE2] SSE;
V	double per[NLAT_2 + NWAY*VSIZE2] SSE;
V	double por[NLAT_2 + NWAY*VSIZE2] SSE;
Q	double rei[NLAT_2 + NWAY*VSIZE2] SSE;
Q	double roi[NLAT_2 + NWAY*VSIZE2] SSE;
V	double tei[NLAT_2 + NWAY*VSIZE2] SSE;
V	double toi[NLAT_2 + NWAY*VSIZE2] SSE;
V	double pei[NLAT_2 + NWAY*VSIZE2] SSE;
V	double poi[NLAT_2 + NWAY*VSIZE2] SSE;

	nk = NLAT_2;	// copy NLAT_2 to a local variable for faster access (inner loop limit)
	#if _GCC_VEC_
	  nk = ((unsigned) nk+(VSIZE2-1))/VSIZE2;
	#endif
	wg = shtns->wg;		ct = shtns->ct;		st = shtns->st;
V	l_2 = shtns->l_2;

	for (k=nk*VSIZE2; k<(nk-1+NWAY)*VSIZE2; ++k) {
Q		rer[k] = 0.0;		ror[k] = 0.0;
V		ter[k] = 0.0;		tor[k] = 0.0;
V		per[k] = 0.0;		por[k] = 0.0;
	}

	if (im == 0) {		// im=0
		alm = shtns->blm;
V		k=0;	do {	// compute symmetric and antisymmetric parts. (do not weight here, it is cheaper to weight y0)
V			double n = creal(Vt[k]);		double s = creal(Vt[NLAT-1-k]);
V			ter[k] = n+s;			tor[k] = n-s;
V		} while(++k < nk*VSIZE2);
V		k=0;	do {	// compute symmetric and antisymmetric parts. (do not weight here, it is cheaper to weight y0)
V			double n = creal(Vp[k]);		double s = creal(Vp[NLAT-1-k]);
V			per[k] = n+s;			por[k] = n-s;
V		} while(++k < nk*VSIZE2);
Q		double r0 = 0.0;
Q		k=0;	do {	// compute symmetric and antisymmetric parts. (do not weight here, it is cheaper to weight y0)
Q			double n = creal(Vr[k]);		double s = creal(Vr[NLAT-1-k]);
Q			rer[k] = n+s;			ror[k] = n-s;
Q			r0 += (n+s)*wg[k];
Q		} while(++k < nk*VSIZE2);
		alm0_rescale = alm[0] * shtns->nphi;	// alm[0] takes into account the fftw normalization, *nphi cancels it
V		Slm[0] = 0.0;		Tlm[0] = 0.0;		// l=0 is zero for the vector transform.
Q		Qlm[0] = r0 * alm0_rescale;			// l=0 is done.
		k = 0;
		for (l=0;l<llim;++l) {
Q			qq[l] = vall(0.0);
V			vw[2*l] = vall(0.0);		vw[2*l+1] = vall(0.0);
		}
		do {
			al = alm;
			rnd cost[NWAY], y0[NWAY], y1[NWAY];
V			rnd sint[NWAY], dy0[NWAY], dy1[NWAY];
Q			rnd rerk[NWAY], rork[NWAY];		// help the compiler to cache into registers.
V			rnd terk[NWAY], tork[NWAY], perk[NWAY], pork[NWAY];
			for (int j=0; j<NWAY; ++j) {
				cost[j] = vread(ct, k+j);
				y0[j] = vall(alm0_rescale) * vread(wg, k+j);		// weight of Gauss quadrature appears here
V				dy0[j] = vall(0.0);
V				sint[j] = -vread(st, k+j);
				y1[j] =  (vall(al[1])*y0[j]) * cost[j];
V				dy1[j] = (vall(al[1])*y0[j]) * sint[j];
Q				rerk[j] = vread(rer, k+j);		rork[j] = vread(ror, k+j);		// cache into registers.
V				terk[j] = vread(ter, k+j);		tork[j] = vread(tor, k+j);
V				perk[j] = vread(per, k+j);		pork[j] = vread(por, k+j);
			}
			al+=2;	l=1;
			while(l<llim) {
				for (int j=0; j<NWAY; ++j) {
V					dy0[j] = vall(al[1])*(cost[j]*dy1[j] + y1[j]*sint[j]) + vall(al[0])*dy0[j];
					y0[j]  = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					qq[l-1]   += y1[j]  * rork[j];
V					vw[2*l-2] += dy1[j] * terk[j];
V					vw[2*l-1] -= dy1[j] * perk[j];
				}
				for (int j=0; j<NWAY; ++j) {
V					dy1[j] = vall(al[3])*(cost[j]*dy0[j] + y0[j]*sint[j]) + vall(al[2])*dy1[j];
					y1[j]  = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					qq[l]     += y0[j]  * rerk[j];
V					vw[2*l]   += dy0[j] * tork[j];
V					vw[2*l+1] -= dy0[j] * pork[j];
				}
				al+=4;	l+=2;
			}
			if (l==llim) {
				for (int j=0; j<NWAY; ++j) {
Q					qq[l-1]   += y1[j]  * rork[j];
V					vw[2*l-2] += dy1[j] * terk[j];
V					vw[2*l-1] -= dy1[j] * perk[j];
				}
			}
			k+=NWAY;
		} while (k < nk);
		for (l=1; l<=llim; ++l) {
			#if _GCC_VEC_
Q				((v2d*)Qlm)[l] = v2d_reduce(qq[l-1], vall(0));
V				((v2d*)Slm)[l] = v2d_reduce(vw[2*l-2], vall(0)) * vdup(l_2[l]);
V				((v2d*)Tlm)[l] = v2d_reduce(vw[2*l-1], vall(0)) * vdup(l_2[l]);
			#else
Q				Qlm[l] = qq[l-1];
V				Slm[l] = vw[2*l-2]*l_2[l];		Tlm[l] = vw[2*l-1]*l_2[l];
			#endif
		}
		#ifdef SHT_VAR_LTR
			for (l=llim+1; l<= LMAX; ++l) {
Q				((v2d*)Qlm)[l] = vdup(0.0);
V				((v2d*)Slm)[l] = vdup(0.0);		((v2d*)Tlm)[l] = vdup(0.0);
			}
		#endif
		
	} else {		// im > 0

		for (k=nk*VSIZE2; k<(nk-1+NWAY)*VSIZE2; ++k) {
Q			rei[k] = 0.0;		roi[k] = 0.0;
V			tei[k] = 0.0;		toi[k] = 0.0;
V			pei[k] = 0.0;		poi[k] = 0.0;
		}

		m = im*MRES;
		l = shtns->tm[im] / VSIZE2;
		alm = shtns->blm + im*(2*(LMAX+1) -m+MRES);
Q		k = ((l*VSIZE2)>>1)*2;		// k must be even here.
Q		do {	// compute symmetric and antisymmetric parts.
3			double sink = st[k];
Q			cplx n = Vr[k];			cplx s = Vr[NLAT-1-k];
3			n *= sink;				s *= sink;
Q			rer[k] = creal(n+s);	rei[k] = cimag(n+s);
Q			ror[k] = creal(n-s);	roi[k] = cimag(n-s);
Q		} while (++k<nk*VSIZE2);
V		k = ((l*VSIZE2)>>1)*2;		// k must be even here.
V		do {	// compute symmetric and antisymmetric parts.
V			cplx n = Vt[k];			cplx s = Vt[NLAT-1-k];
V			ter[k] = creal(n+s);	tei[k] = cimag(n+s);
V			tor[k] = creal(n-s);	toi[k] = cimag(n-s);
V		} while (++k<nk*VSIZE2);
V		k = ((l*VSIZE2)>>1)*2;		// k must be even here.
V		do {	// compute symmetric and antisymmetric parts.
V			cplx n = Vp[k];			cplx s = Vp[NLAT-1-k];
V			per[k] = creal(n+s);	pei[k] = cimag(n+s);
V			por[k] = creal(n-s);	poi[k] = cimag(n-s);
V		} while (++k<nk*VSIZE2);

		k=l;
		#if _GCC_VEC_
Q			rnd* q = qq;
V			rnd* v = vw;
		#else
Q			double* q = (double *) qq;	//Qlm;
V			double* v = (double *) vw;
V			double* t = (double *) vw;
		#endif
		for (l=llim-m; l>=0; l--) {
Q			q[0] = vall(0.0);		q[1] = vall(0.0);		q+=2;
V			v[0] = vall(0.0);		v[1] = vall(0.0);
V			v[2] = vall(0.0);		v[3] = vall(0.0);		v+=4;
		}
		alm0_rescale = alm[0] * (shtns->nphi*2);
		do {
		#if _GCC_VEC_
Q			rnd* q = qq;
V			rnd* v = vw;
		#else
Q			double* q = (double *) qq;	//Qlm;
V			double* v = (double *) vw;
V			double* t = (double *) vw;
		#endif
			al = alm;
			rnd cost[NWAY], y0[NWAY], y1[NWAY];
Q			rnd rerk[NWAY], reik[NWAY], rork[NWAY], roik[NWAY];		// help the compiler to cache into registers.
V			rnd terk[NWAY], teik[NWAY], tork[NWAY], toik[NWAY];
V			rnd perk[NWAY], peik[NWAY], pork[NWAY], poik[NWAY];
			for (int j=0; j<NWAY; ++j) {
				cost[j] = vread(st, k+j);
				y0[j] = vall(0.5);
			}
Q			l=m;
V			l=m-1;
			long int ny = 0;	// exponent to extend double precision range.
		if ((int)llim <= SHT_L_RESCALE_FLY) {
			do {		// sin(theta)^m
				if (l&1) for (int j=0; j<NWAY; ++j) y0[j] *= cost[j];
				for (int j=0; j<NWAY; ++j) cost[j] *= cost[j];
			} while(l >>= 1);
		} else {
			long int nsint = 0;
			do {		// sin(theta)^m		(use rescaling to avoid underflow)
				if (l&1) {
					for (int j=NWAY-1; j>=0; --j) y0[j] *= cost[j];
					ny += nsint;
					if (vlo(y0[NWAY-1]) < (SHT_ACCURACY+1.0/SHT_SCALE_FACTOR)) {
						ny--;
						for (int j=NWAY-1; j>=0; --j) y0[j] *= vall(SHT_SCALE_FACTOR);
					}
				}
				for (int j=NWAY-1; j>=0; --j) cost[j] *= cost[j];
				nsint += nsint;
				if (vlo(cost[NWAY-1]) < 1.0/SHT_SCALE_FACTOR) {
					nsint--;
					for (int j=NWAY-1; j>=0; --j) cost[j] *= vall(SHT_SCALE_FACTOR);
				}
			} while(l >>= 1);
		}
			for (int j=0; j<NWAY; ++j) {
				y0[j] *= vall(alm0_rescale);
				cost[j] = vread(ct, k+j);
				y1[j]  = (vall(al[1])*y0[j]) *cost[j];
			}
			l=m;	al+=2;
			while ((ny<0) && (l<llim)) {		// ylm treated as zero and ignored if ny < 0
				for (int j=0; j<NWAY; ++j) {
					y0[j] = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
					y1[j] = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				l+=2;	al+=4;
				if (fabs(vlo(y0[NWAY-1])) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0) {		// rescale when value is significant
					++ny;
					for (int j=0; j<NWAY; ++j) {
						y0[j] *= vall(1.0/SHT_SCALE_FACTOR);		y1[j] *= vall(1.0/SHT_SCALE_FACTOR);
					}
				}
			}
		  if (ny == 0) {
Q			q+=2*(l-m);
V			v+=4*(l-m);
			for (int j=0; j<NWAY; ++j) {	// prefetch
				y0[j] *= vread(wg, k+j);		y1[j] *= vread(wg, k+j);		// weight appears here (must be after the previous accuracy loop).
Q				rerk[j] = vread( rer, k+j);		reik[j] = vread( rei, k+j);		rork[j] = vread( ror, k+j);		roik[j] = vread( roi, k+j);
V				terk[j] = vread( ter, k+j);		teik[j] = vread( tei, k+j);		tork[j] = vread( tor, k+j);		toik[j] = vread( toi, k+j);
V				perk[j] = vread( per, k+j);		peik[j] = vread( pei, k+j);		pork[j] = vread( por, k+j);		poik[j] = vread( poi, k+j);
			}
			while (l<llim) {	// compute even and odd parts
Q				for (int j=0; j<NWAY; ++j)	{	q[0] += y0[j] * rerk[j];	q[1] += y0[j] * reik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[0] += y0[j] * terk[j];	v[1] += y0[j] * teik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[2] += y0[j] * perk[j];	v[3] += y0[j] * peik[j];	}
				for (int j=0; j<NWAY; ++j) {
					y0[j] = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
Q				for (int j=0; j<NWAY; ++j)	{	q[2] += y1[j] * rork[j];	q[3] += y1[j] * roik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[4] += y1[j] * tork[j];	v[5] += y1[j] * toik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[6] += y1[j] * pork[j];	v[7] += y1[j] * poik[j];	}
Q				q+=4;
V				v+=8;
				for (int j=0; j<NWAY; ++j) {
					y1[j] = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				l+=2;	al+=4;
			}
V				for (int j=0; j<NWAY; ++j)	{	v[0] += y0[j] * terk[j];	v[1] += y0[j] * teik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[2] += y0[j] * perk[j];	v[3] += y0[j] * peik[j];	}
			if (l==llim) {
Q				for (int j=0; j<NWAY; ++j)	{	q[0] += y0[j] * rerk[j];	q[1] += y0[j] * reik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[4] += y1[j] * tork[j];	v[5] += y1[j] * toik[j];	}
V				for (int j=0; j<NWAY; ++j)	{	v[6] += y1[j] * pork[j];	v[7] += y1[j] * poik[j];	}
			}
		  }
			k+=NWAY;
		} while (k < nk);

Q		#if _GCC_VEC_
Q			for (l=0; l<=llim-m; ++l) {
Q				((v2d*)Qlm)[l] = v2d_reduce(qq[2*l], qq[2*l+1]);
Q			}
Q		#else
Q			for (l=0; l<=llim-m; ++l) {
Q				Qlm[l] = qq[2*l] + I*qq[2*l+1];
Q			}
Q		#endif

V		{	// convert from the two scalar SH to vector SH
V			// Slm = - (I*m*Wlm + MX*Vlm) / (l*(l+1))		=> why does this work ??? (aliasing of 1/sin(theta) ???)
V			// Tlm = - (I*m*Vlm - MX*Wlm) / (l*(l+1))
V			double* mx = shtns->mx_van + 2*LM(shtns,m,m);	//(im*(2*(LMAX+1)-(m+MRES))) + 2*m;
V			s2d em = vdup(m);
V			v2d vl = v2d_reduce(vw[0], vw[1]);
V			v2d wl = v2d_reduce(vw[2], vw[3]);
V			v2d sl = vdup( 0.0 );
V			v2d tl = vdup( 0.0 );
V			for (int l=0; l<=llim-m; l++) {
V				s2d mxu = vdup( mx[2*l] );
V				s2d mxl = vdup( mx[2*l+1] );		// mxl for next iteration
V				sl = addi( sl ,  em*wl );
V				tl = addi( tl ,  em*vl );
V				v2d sl1 =  mxl*vl;			// vs for next iter
V				v2d tl1 = -mxl*wl;			// wt for next iter
V				vl = v2d_reduce(vw[4*l+4], vw[4*l+5]);		// kept for next iteration
V				wl = v2d_reduce(vw[4*l+6], vw[4*l+7]);
V				sl += mxu*vl;
V				tl -= mxu*wl;
V				((v2d*)Slm)[l] = -sl * vdup(l_2[l+m]);
V				((v2d*)Tlm)[l] = -tl * vdup(l_2[l+m]);
V				sl = sl1;
V				tl = tl1;
V			}
V		}

		#ifdef SHT_VAR_LTR
			for (l=llim+1-m; l<=LMAX-m; ++l) {
Q				((v2d*)Qlm)[l] = vdup(0.0);
V				((v2d*)Slm)[l] = vdup(0.0);		((v2d*)Tlm)[l] = vdup(0.0);
			}
		#endif
	}

  }

  #endif
