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
# it is intended for "make" to generate C code for similar SHT functions,
# from one generic function + tags.
# > See Makefile and SHT.c
# Basically, there are tags at the beginning of lines that are information
# to keep or remove the line depending on the function to build.
# tags :
# Q : line for scalar transform
# V : line for vector transform (both spheroidal and toroidal)
# S : line for vector transfrom, spheroidal component
# T : line for vector transform, toroidal component.

	static
3	void GEN3(_sy3,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, v2d *BrF, v2d *BtF, v2d *BpF, const long int llim, const int imlim) {
QX	void GEN3(_sy1,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Qlm, v2d *BrF, long int llim, const int imlim) {
  #ifndef SHT_GRAD
VX	void GEN3(_sy2,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Slm, cplx *Tlm, v2d *BtF, v2d *BpF, const long int llim, const int imlim) {
  #else
S	void GEN3(_sy1s,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Slm, v2d *BtF, v2d *BpF, const long int llim, const int imlim) {
T	void GEN3(_sy1t,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Tlm, v2d *BtF, v2d *BpF, const long int llim, const int imlim) {
  #endif

  #ifndef SHT_AXISYM
Q	#define qr(l) vall(creal(Ql[l]))
Q	#define qi(l) vall(cimag(Ql[l]))
V	#define vr(l) vall( ((double*) VWl)[4*(l)]   )
V	#define vi(l) vall( ((double*) VWl)[4*(l)+1] )
V	#define wr(l) vall( ((double*) VWl)[4*(l)+2] )
V	#define wi(l) vall( ((double*) VWl)[4*(l)+3] )
	unsigned im;
  #endif
	unsigned m0, mstep;
	long int nk,k,l,m;
	double *alm, *al;
	double *ct, *st;
QX	double Ql0[llim+2];
V	v2d VWl[llim*2+4];

	ct = shtns->ct;		st = shtns->st;
	nk = NLAT_2;
	#if _GCC_VEC_
		nk = ((unsigned)(nk+VSIZE2-1)) / VSIZE2;
	#endif

	#ifndef _OPENMP
		m0 = 0;		mstep = 1;
	#else
		m0 = omp_get_thread_num();
		mstep = omp_get_num_threads();
		if (m0 == 0)
	#endif
	{	//	im=0;
S		double* const Sl0 = (double*) VWl;
T		double* const Tl0 = (double*) VWl + llim+2;
3		double* const Ql0 = (double*) (VWl + llim+2);
		#ifdef SHT_GRAD
		  #ifndef SHT_AXISYM
			#ifdef _GCC_VEC_
S				k=0; do { BpF[k]=vdup(0.0); } while(++k<NLAT_2);
T				k=0; do { BtF[k]=vdup(0.0); } while(++k<NLAT_2);
			#else
S				k=0; do { BpF[k]=vdup(0.0); } while(++k<NLAT);
T				k=0; do { BtF[k]=vdup(0.0); } while(++k<NLAT);
			#endif
		  #else
S			if (BpF != NULL) { int k=0; do { BpF[k]=vdup(0.0); } while(++k<NLAT_2); }
T			if (BtF != NULL) { int k=0; do { BtF[k]=vdup(0.0); } while(++k<NLAT_2); }
		  #endif
		#endif
 		l=1;
		alm = shtns->alm;
Q		Ql0[0] = (double) Qlm[0];		// l=0
		do {		// for m=0, compress the complex Q,S,T to double
Q			Ql0[l] = (double) Qlm[l];	//	Ql[l+1] = (double) Qlm[l+1];
S			Sl0[l-1] = (double) Slm[l];	//	Sl[l] = (double) Slm[l+1];
T			Tl0[l-1] = (double) Tlm[l];	//	Tl[l] = (double) Tlm[l+1];
			++l;
		} while(l<=llim);
		k=0;
		do {
			l=0;	al = alm;
			rnd cost[NWAY], y0[NWAY], y1[NWAY];
V			rnd sint[NWAY], dy0[NWAY], dy1[NWAY];
Q			rnd re[NWAY], ro[NWAY];
S			rnd te[NWAY], to[NWAY];
T			rnd pe[NWAY], po[NWAY];
			for (int j=0; j<NWAY; ++j) {
				cost[j] = vread(ct, j+k);
V				sint[j] = -vread(st, j+k);
V			#ifdef SHTNS4MAGIC
V				sint[j] *= -sint[j];
V			#endif
				y0[j] = vall(al[0]);
V				dy0[j] = vall(0.0);
Q				re[j] = y0[j] * vall(Ql0[0]);
S				to[j] = dy0[j];
T				po[j] = dy0[j];
			}
			for (int j=0; j<NWAY; ++j) {
				y1[j]  = vall(al[0]*al[1]) * cost[j];
V				dy1[j] = vall(al[0]*al[1]) * sint[j];
			}
			for (int j=0; j<NWAY; ++j) {
Q				ro[j] = y1[j] * vall(Ql0[1]);
S				te[j] = dy1[j] * vall(Sl0[0]);
T				pe[j] = -dy1[j] * vall(Tl0[0]);
			}
			al+=2;	l+=2;
			while(l<llim) {
				for (int j=0; j<NWAY; ++j) {
V					dy0[j] = vall(al[1])*(cost[j]*dy1[j] + y1[j]*sint[j]) + vall(al[0])*dy0[j];
					y0[j]  = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					re[j] += y0[j] * vall(Ql0[l]);
S					to[j] += dy0[j] * vall(Sl0[l-1]);
T					po[j] -= dy0[j] * vall(Tl0[l-1]);
				}
				for (int j=0; j<NWAY; ++j) {
V					dy1[j] = vall(al[3])*(cost[j]*dy0[j] + y0[j]*sint[j]) + vall(al[2])*dy1[j];
					y1[j]  = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					ro[j] += y1[j] * vall(Ql0[l+1]);
S					te[j] += dy1[j] * vall(Sl0[l]);
T					pe[j] -= dy1[j] * vall(Tl0[l]);
				}
				al+=4;	l+=2;
			}
			if (l==llim) {
				for (int j=0; j<NWAY; ++j) {
V					dy0[j] = vall(al[1])*(cost[j]*dy1[j] + y1[j]*sint[j]) + vall(al[0])*dy0[j];
					y0[j]  = vall(al[1])*cost[j]*y1[j] + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
Q					re[j] += y0[j] * vall(Ql0[l]);
S					to[j] += dy0[j] * vall(Sl0[l-1]);
T					po[j] -= dy0[j] * vall(Tl0[l-1]);
				}
			}
		#ifndef SHT_AXISYM
Q			v2d* const Br0 = BrF;
S			v2d* const Bt0 = BtF;
T			v2d* const Bp0 = BpF;
		#else
Q			double* const Br0 = (double*) BrF;
S			double* const Bt0 = (double*) BtF;
T			double* const Bp0 = (double*) BpF;
		#endif
		#ifndef SHTNS4MAGIC
			for (int j=0; j<NWAY; ++j) {
Q				S2D_STORE(Br0, j+k, re[j], ro[j])
S				S2D_STORE(Bt0, j+k, te[j], to[j])
T				S2D_STORE(Bp0, j+k, pe[j], po[j])
			}
		#else
			for (int j=0; j<NWAY; ++j) {
				if ((k+j)>=nk) break;
Q				S2D_STORE_4MAGIC(Br0, j+k, re[j], ro[j])
S				S2D_STORE_4MAGIC(Bt0, j+k, te[j], to[j])
T				S2D_STORE_4MAGIC(Bp0, j+k, pe[j], po[j])
			}
		#endif
			k+=NWAY;
		} while (k < nk);
		m0=mstep;
	}

  #ifndef SHT_AXISYM
	#if _GCC_VEC_
Q		BrF += m0*NLAT_2;
V		BtF += m0*NLAT_2;	BpF += m0*NLAT_2;
	#else
Q		BrF += m0*NLAT;
V		BtF += m0*NLAT;		BpF += m0*NLAT;
	#endif
	for (im=m0; im<imlim; im+=mstep) {
		m = im*MRES;
		//l = LiM(shtns, 0,im);
		l = (im*(2*(LMAX+1)-(m+MRES)))>>1;
		//alm = shtns->alm[im];
		alm = shtns->alm + 2*(l+m);		// shtns->alm + im*(2*(LMAX+1) -m+MRES);

Q		cplx* Ql = &Qlm[l];	// virtual pointer for l=0 and im
V		{	// convert from vector SH to scalar SH
V			// Vlm =  st*d(Slm)/dtheta + I*m*Tlm
V			// Wlm = -st*d(Tlm)/dtheta + I*m*Slm
V			// store interleaved: VWlm(2*l) = Vlm(l);	VWlm(2*l+1) = Vlm(l);
V			double* mx = shtns->mx_stdt + 2*l;
S			v2d* Sl = (v2d*) &Slm[l];	// virtual pointer for l=0 and im
T			v2d* Tl = (v2d*) &Tlm[l];
V			s2d em = vdup(m);
S			v2d sl = Sl[m];
T			v2d tl = Tl[m];
V			v2d vs = vdup( 0.0 );
V			v2d wt = vdup( 0.0 );
V			for (int l=m; l<=llim; l++) {
V				s2d mxu = vdup( mx[2*l] );
V				s2d mxl = vdup( mx[2*l+1] );		// mxl for next iteration
T				vs = addi( vs ,  em*tl );
S				wt = addi( wt ,  em*sl );
S				v2d vs1 = mxl*sl;			// vs for next iter
T				v2d wt1 = -mxl*tl;			// wt for next iter
V				if (l<llim) {
S					sl = Sl[l+1];		// kept for next iteration
T					tl = Tl[l+1];
S					vs += mxu*sl;
T					wt -= mxu*tl;
V				}
V				VWl[2*l]   = vs;
V				VWl[2*l+1] = wt;
V				vs = vdup( 0.0 );		wt = vdup( 0.0 );
S				vs = vs1;
T				wt = wt1;
V			}
V			VWl[2*llim+2]   = vs;
V			VWl[2*llim+3] = wt;
V		}

		k=0;	l=shtns->tm[im];
	#if _GCC_VEC_
		l>>=1;		// stay on a 16 byte boundary
		while (k<l) {	// polar optimization
		  #ifndef SHTNS4MAGIC
Q			BrF[k] = vdup(0.0);				BrF[(NPHI-2*im)*NLAT_2 + k] = vdup(0.0);
Q			BrF[NLAT_2-l+k] = vdup(0.0);	BrF[(NPHI+1-2*im)*NLAT_2 -l+k] = vdup(0.0);
V			BtF[k] = vdup(0.0);				BtF[(NPHI-2*im)*NLAT_2 + k] = vdup(0.0);
V			BtF[NLAT_2-l+k] = vdup(0.0);	BtF[(NPHI+1-2*im)*NLAT_2 -l+k] = vdup(0.0);
V			BpF[k] = vdup(0.0);				BpF[(NPHI-2*im)*NLAT_2 + k] = vdup(0.0);
V			BpF[NLAT_2-l+k] = vdup(0.0);	BpF[(NPHI+1-2*im)*NLAT_2 -l+k] = vdup(0.0);
		  #else
Q			BrF[2*k] = vdup(0.0);			BrF[(NPHI-2*im)*NLAT_2 + 2*k] = vdup(0.0);
Q			BrF[2*k+1] = vdup(0.0);			BrF[(NPHI-2*im)*NLAT_2 +2*k+1] = vdup(0.0);
V			BtF[2*k] = vdup(0.0);			BtF[(NPHI-2*im)*NLAT_2 + 2*k] = vdup(0.0);
V			BtF[2*k+1] = vdup(0.0);			BtF[(NPHI-2*im)*NLAT_2 +2*k+1] = vdup(0.0);
V			BpF[2*k] = vdup(0.0);			BpF[(NPHI-2*im)*NLAT_2 + 2*k] = vdup(0.0);
V			BpF[2*k+1] = vdup(0.0);			BpF[(NPHI-2*im)*NLAT_2 +2*k+1] = vdup(0.0);
		  #endif
			++k;
		}
		k = ((unsigned) k) / (VSIZE2/2);
	#else
		while (k<l) {	// polar optimization
		  #ifndef SHTNS4MAGIC
Q			BrF[k] = 0.0;		BrF[NLAT-l+k] = 0.0;
V			BtF[k] = 0.0;		BtF[NLAT-l+k] = 0.0;
V			BpF[k] = 0.0;		BpF[NLAT-l+k] = 0.0;
		  #else
Q			BrF[2*k] = 0.0;		BrF[2*k+1] = 0.0;
V			BtF[2*k] = 0.0;		BtF[2*k+1] = 0.0;
V			BpF[2*k] = 0.0;		BpF[2*k+1] = 0.0;
		  #endif
			++k;
		}
	#endif
		do {
			al = alm;
			rnd cost[NWAY], y0[NWAY], y1[NWAY];
Q			rnd rer[NWAY], rei[NWAY], ror[NWAY], roi[NWAY];
V			rnd ter[NWAY], tei[NWAY], tor[NWAY], toi[NWAY];
V			rnd per[NWAY], pei[NWAY], por[NWAY], poi[NWAY];
			for (int j=0; j<NWAY; ++j) {
				cost[j] = vread(st, k+j);
				y0[j] = vall(1.0);
			}
Q			l=m;
V		#ifndef SHTNS4MAGIC
V			l=m-1;
V		#else
V			l=m;
V		#endif
			long int ny = 0;
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
				cost[j] = vread(ct, j+k);
Q				ror[j] = vall(0.0);		roi[j] = vall(0.0);
Q				rer[j] = vall(0.0);		rei[j] = vall(0.0);
			}
			for (int j=0; j<NWAY; ++j) {
				y1[j]  = (vall(al[1])*y0[j]) *cost[j];		//	y1[j] = vall(al[1])*cost[j]*y0[j];
V				por[j] = vall(0.0);		tei[j] = vall(0.0);
V				tor[j] = vall(0.0);		pei[j] = vall(0.0);
V				poi[j] = vall(0.0);		ter[j] = vall(0.0);
V				toi[j] = vall(0.0);		per[j] = vall(0.0);
			}
			l=m;		al+=2;
			while ((ny<0) && (l<llim)) {		// ylm treated as zero and ignored if ny < 0
				for (int j=0; j<NWAY; ++j) {
					y0[j] = (vall(al[1])*cost[j])*y1[j] + vall(al[0])*y0[j];
				}
				for (int j=0; j<NWAY; ++j) {
					y1[j] = (vall(al[3])*cost[j])*y0[j] + vall(al[2])*y1[j];
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
			while (l<llim) {	// compute even and odd parts
Q				for (int j=0; j<NWAY; ++j) {	rer[j] += y0[j]  * qr(l);		rei[j] += y0[j] * qi(l);	}
V				for (int j=0; j<NWAY; ++j) {	ter[j] += y0[j]  * vr(l);		tei[j] += y0[j] * vi(l);	}
V				for (int j=0; j<NWAY; ++j) {	per[j] += y0[j]  * wr(l);		pei[j] += y0[j] * wi(l);	}
				for (int j=0; j<NWAY; ++j) {
					y0[j] = vall(al[1])*(cost[j]*y1[j]) + vall(al[0])*y0[j];
				}
Q				for (int j=0; j<NWAY; ++j) {	ror[j] += y1[j]  * qr(l+1);		roi[j] += y1[j] * qi(l+1);	}
V				for (int j=0; j<NWAY; ++j) {	tor[j] += y1[j]  * vr(l+1);		toi[j] += y1[j] * vi(l+1);	}
V				for (int j=0; j<NWAY; ++j) {	por[j] += y1[j]  * wr(l+1);		poi[j] += y1[j] * wi(l+1);	}
				for (int j=0; j<NWAY; ++j) {
					y1[j] = vall(al[3])*(cost[j]*y0[j]) + vall(al[2])*y1[j];
				}
				l+=2;	al+=4;
			}
V				for (int j=0; j<NWAY; ++j) {	ter[j] += y0[j]  * vr(l);		tei[j] += y0[j] * vi(l);	}
V				for (int j=0; j<NWAY; ++j) {	per[j] += y0[j]  * wr(l);		pei[j] += y0[j] * wi(l);	}
			if (l==llim) {
Q				for (int j=0; j<NWAY; ++j) {	rer[j] += y0[j]  * qr(l);		rei[j] += y0[j] * qi(l);	}
V				for (int j=0; j<NWAY; ++j) {	tor[j] += y1[j]  * vr(l+1);		toi[j] += y1[j] * vi(l+1);	}
V				for (int j=0; j<NWAY; ++j) {	por[j] += y1[j]  * wr(l+1);		poi[j] += y1[j] * wi(l+1);	}
			}
3		#ifndef SHTNS4MAGIC
3			for (int j=0; j<NWAY; ++j) cost[j]  = vread(st, k+j);
3			for (int j=0; j<NWAY; ++j) {  rer[j] *= cost[j];  ror[j] *= cost[j];	rei[j] *= cost[j];  roi[j] *= cost[j];  }
3		#endif
		  }
		#ifndef SHTNS4MAGIC
			for (int j=0; j<NWAY; ++j) {
Q				S2D_CSTORE(BrF, k+j, rer[j], ror[j], rei[j], roi[j])
V				S2D_CSTORE(BtF, k+j, ter[j], tor[j], tei[j], toi[j])
V				S2D_CSTORE(BpF, k+j, per[j], por[j], pei[j], poi[j])
			}
		#else
			for (int j=0; j<NWAY; ++j) {
				if ((k+j)>=nk) break;
Q				S2D_CSTORE_4MAGIC(BrF, k+j, rer[j], ror[j], rei[j], roi[j])
V				S2D_CSTORE_4MAGIC(BtF, k+j, ter[j], tor[j], tei[j], toi[j])
V				S2D_CSTORE_4MAGIC(BpF, k+j, per[j], por[j], pei[j], poi[j])
			}
		#endif
			k+=NWAY;
		} while (k < nk);
	#if _GCC_VEC_
Q		BrF += mstep*NLAT_2;
V		BtF += mstep*NLAT_2;	BpF += mstep*NLAT_2;
	#else
Q		BrF += mstep*NLAT;
V		BtF += mstep*NLAT;	BpF += mstep*NLAT;
	#endif
	}

	#if _GCC_VEC_
	while(im <= NPHI-imlim) {	// padding for high m's
		k=0;
		do {
Q			BrF[k] = vdup(0.0);
V			BtF[k] = vdup(0.0);		BpF[k] = vdup(0.0);
		} while (++k < NLAT_2);
Q		BrF += mstep*NLAT_2;
V		BtF += mstep*NLAT_2;	BpF += mstep*NLAT_2;
	  im+=mstep;
	}
	#else
	while(im <= NPHI/2) {	// padding for high m's
		k=0;
		do {
Q			BrF[k] = 0.0;
V			BtF[k] = 0.0;	BpF[k] = 0.0;
		} while (++k < NLAT);
Q		BrF += mstep*NLAT;
V		BtF += mstep*NLAT;	BpF += mstep*NLAT;
	  im+=mstep;
	}
	#endif
  #endif
}

Q	#undef qr
Q	#undef qi
S	#undef sr
S	#undef si
T	#undef tr
T	#undef ti

3	static void GEN3(SHqst_to_spat_omp,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, long int llim) {
QX	static void GEN3(SH_to_spat_omp,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Qlm, double *Vr, long int llim) {
  #ifndef SHT_GRAD
VX	static void GEN3(SHsphtor_to_spat_omp,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, long int llim) {
  #else
S	static void GEN3(SHsph_to_spat_omp,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Slm, double *Vt, double *Vp, long int llim) {
T	static void GEN3(SHtor_to_spat_omp,NWAY,SUFFIX)(shtns_cfg shtns, cplx *Tlm, double *Vt, double *Vp, long int llim) {
  #endif

	int k;
	unsigned imlim = 0;
Q	v2d* BrF = (v2d*) Vr;
V	v2d* BtF = (v2d*) Vt;	v2d* BpF = (v2d*) Vp;

  #ifndef SHT_AXISYM
	imlim = MTR;
	#ifdef SHT_VAR_LTR
		if (imlim*MRES > (unsigned) llim) imlim = ((unsigned) llim)/MRES;		// 32bit mul and div should be faster
	#endif
	#ifdef _GCC_VEC_
	if (shtns->fftc_mode > 0) {		// alloc memory for the FFT
		unsigned long nv = shtns->nspat;
QX		BrF = (v2d*) VMALLOC( nv * sizeof(double) );
VX		BtF = (v2d*) VMALLOC( 2*nv * sizeof(double) );
VX		BpF = BtF + nv/2;
3		BrF = (v2d*) VMALLOC( 3*nv * sizeof(double) );
3		BtF = BrF + nv/2;		BpF = BrF + nv;
	}
	#else
	if (shtns->ncplx_fft > 0) {		// alloc memory for the FFT
QX		BrF = VMALLOC( shtns->ncplx_fft * sizeof(cplx) );
VX		BtF = VMALLOC( 2* shtns->ncplx_fft * sizeof(cplx) );
VX		BpF = BtF + shtns->ncplx_fft;
3		BrF = VMALLOC( 3* shtns->ncplx_fft * sizeof(cplx) );
3		BtF = BrF + shtns->ncplx_fft;		BpF = BtF + shtns->ncplx_fft;
	}
	#endif
  #endif
	imlim += 1;
  
  #pragma omp parallel num_threads(shtns->nthreads)
  {
3	GEN3(_sy3,NWAY,SUFFIX)(shtns, Qlm, Slm, Tlm, BrF, BtF, BpF, llim, imlim);
QX	GEN3(_sy1,NWAY,SUFFIX)(shtns, Qlm, BrF, llim, imlim);
	#ifndef SHT_GRAD
VX		GEN3(_sy2,NWAY,SUFFIX)(shtns, Slm, Tlm, BtF, BpF, llim, imlim);
	#else
S		GEN3(_sy1s,NWAY,SUFFIX)(shtns, Slm, BtF, BpF, llim, imlim);
T		GEN3(_sy1t,NWAY,SUFFIX)(shtns, Tlm, BtF, BpF, llim, imlim);
	#endif

  #ifndef SHT_AXISYM
V	#ifndef HAVE_LIBFFTW3_OMP
V	  #pragma omp barrier
V	  #if _GCC_VEC_
V		if (shtns->fftc_mode == 0) {
3			#pragma omp single nowait
3			fftw_execute_dft(shtns->ifftc, (cplx *) BrF, (cplx *) Vr);
V			#pragma omp single nowait
V			fftw_execute_dft(shtns->ifftc, (cplx *) BtF, (cplx *) Vt);
V			#pragma omp single nowait
V			fftw_execute_dft(shtns->ifftc, (cplx *) BpF, (cplx *) Vp);
V		} else if (shtns->fftc_mode > 0) {		// split dft
3			#pragma omp single nowait
3			fftw_execute_split_dft(shtns->ifftc,((double*)BrF)+1, ((double*)BrF), Vr+NPHI, Vr);
V			#pragma omp single nowait
V			fftw_execute_split_dft(shtns->ifftc,((double*)BtF)+1, ((double*)BtF), Vt+NPHI, Vt);
V			#pragma omp single nowait
V			fftw_execute_split_dft(shtns->ifftc,((double*)BpF)+1, ((double*)BpF), Vp+NPHI, Vp);
V		}
V	  #else
3		#pragma omp single nowait
3		fftw_execute_dft_c2r(shtns->ifft, (cplx *) BrF, Vr);
V		#pragma omp single nowait
V		fftw_execute_dft_c2r(shtns->ifft, (cplx *) BtF, Vt);
V		#pragma omp single nowait
V		fftw_execute_dft_c2r(shtns->ifft, (cplx *) BpF, Vp);
V	  #endif
V	#endif
  #endif

  }

  #ifndef SHT_AXISYM
    // NPHI > 1 as SHT_AXISYM is not defined.
	#if _GCC_VEC_
  	if (shtns->fftc_mode >= 0) {
		if (shtns->fftc_mode == 0) {
V		  #ifdef HAVE_LIBFFTW3_OMP
Q			fftw_execute_dft(shtns->ifftc, (cplx *) BrF, (cplx *) Vr);
V			fftw_execute_dft(shtns->ifftc, (cplx *) BtF, (cplx *) Vt);
V			fftw_execute_dft(shtns->ifftc, (cplx *) BpF, (cplx *) Vp);
V		  #endif
		} else {		// split dft
V		  #ifdef HAVE_LIBFFTW3_OMP
Q			fftw_execute_split_dft(shtns->ifftc,((double*)BrF)+1, ((double*)BrF), Vr+NPHI, Vr);
V			fftw_execute_split_dft(shtns->ifftc,((double*)BtF)+1, ((double*)BtF), Vt+NPHI, Vt);
V			fftw_execute_split_dft(shtns->ifftc,((double*)BpF)+1, ((double*)BpF), Vp+NPHI, Vp);
V		  #endif
Q			VFREE(BrF);
VX			VFREE(BtF);		// this frees also BpF.
		}
	}
	#else
	if (shtns->ncplx_fft >= 0) {
V	  #ifdef HAVE_LIBFFTW3_OMP
Q		fftw_execute_dft_c2r(shtns->ifft, (cplx *) BrF, Vr);
V		fftw_execute_dft_c2r(shtns->ifft, (cplx *) BtF, Vt);
V		fftw_execute_dft_c2r(shtns->ifft, (cplx *) BpF, Vp);
V	  #endif
		if (shtns->ncplx_fft > 0) {		// free memory
Q			VFREE(BrF);
VX			VFREE(BtF);		// this frees also BpF.
		}
	}
	#endif
  #endif

  }
