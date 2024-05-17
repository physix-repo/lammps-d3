// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: 
------------------------------------------------------------------------- */

#include "pair_dftd3.h"

#include <iostream>

#include <cmath>
#include <array>

#include "atom.h"
#include "force.h"

#include "my_page.h"
#include "neigh_list.h"
#include "neighbor.h"

#include "comm.h"
#include "memory.h"
#include "error.h"

#include "potential_file_reader.h"


/* ---------------------------------------------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------------------------------------------- */

// global ad hoc parameters
static constexpr double K1 = 16.0;
static constexpr double K2 = 4.0/3.0;
static constexpr double K3 = -4.0;

/*  reasonable choices for k3 are between 3 and 5
    this gives smoth curves with maxima around the integer values
    k3=3 give for CN=0 a slightly smaller value than computed
    for the free atom. This also yields to larger CN for atoms
    in larger molecules but with the same chemical environment
    which is physically not right.
    values >5 might lead to bumps in the potential.
*/

static constexpr double cn_thr = 100;  // 20*20 Bohr^2 ; 20 Bohr = 10.58 Angstrom  : 400
//static constexpr double cutoff = 1400;

static constexpr int NPARAMS_PER_LINE = 5;
static constexpr int NUM_ELEMENTS = 94;

/* ---------------------------------------------------------------------------------------------------------- */


// these new data are scaled with k2=4.0/3.0  and converted via
// autoang=0.52917726 :           rcov=k2*rcov/autoang

std::array<double, NUM_ELEMENTS> rcov = {
    0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
    1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
    3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
    2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
    3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
    2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
    2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
    2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
    3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
    2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
    3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
    4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
    3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
    3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
    3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
    2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
    3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
    3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
    3.82984466, 3.85504098, 3.88023730, 3.90543362
};

//  r2r4 = sqrt(0.5*r2r4(i)*dfloat(i)**0.5 ) with i=elementnumber
//  the large number of digits is just to keep the results consistent
//  with older versions.

std::array<double, NUM_ELEMENTS> r2r4 = {
    2.00734898,  1.56637132,  5.01986934,  3.85379032,  3.64446594,
    3.10492822,  2.71175247,  2.59361680,  2.38825250,  2.21522516,
    6.58585536,  5.46295967,  5.65216669,  4.88284902,  4.29727576,
    4.04108902,  3.72932356,  3.44677275,  7.97762753,  7.07623947,
    6.60844053,  6.28791364,  6.07728703,  5.54643096,  5.80491167,
    5.58415602,  5.41374528,  5.28497229,  5.22592821,  5.09817141,
    6.12149689,  5.54083734,  5.06696878,  4.87005108,  4.59089647,
    4.31176304,  9.55461698,  8.67396077,  7.97210197,  7.43439917,
    6.58711862,  6.19536215,  6.01517290,  5.81623410,  5.65710424,
    5.52640661,  5.44263305,  5.58285373,  7.02081898,  6.46815523,
    5.98089120,  5.81686657,  5.53321815,  5.25477007, 11.02204549,
    10.15679528, 9.35167836,  9.06926079,  8.97241155,  8.90092807,
    8.85984840,  8.81736827,  8.79317710,  7.89969626,  8.80588454,
    8.42439218,  8.54289262,  8.47583370,  8.45090888,  8.47339339,
    7.83525634,  8.20702843,  7.70559063,  7.32755997,  7.03887381,
    6.68978720,  6.05450052,  5.88752022,  5.70661499,  5.78450695,
    7.79780729,  7.26443867,  6.78151984,  6.67883169,  6.39024318,
    6.09527958, 11.79156076, 11.10997644,  9.51377795,  8.67197068,
    8.77140725,  8.65402716,  8.53923501,  8.85024712
};

/* ---------------------------------------------------------------------------------------------------------- */

PairDFTD3::PairDFTD3(LAMMPS *lmp) : Pair(lmp)
{
  nmax = 0;
  NCo = nullptr;
  map = nullptr;

  ipage = nullptr;
  pgsize = oneatom = 0;

  r0ab = nullptr;

  comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

PairDFTD3::~PairDFTD3()
{
  memory->destroy(NCo);
  memory->destroy(map);

  delete[] ipage;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);

    memory->destroy(r0ab);
  }
}

/* ---------------------------------------------------------------------------------------------------------- */

void PairDFTD3::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int sht_jnum, *sht_jlist, nj;

  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair,factor_lj;
  double rsq,r,rr,dexp,r2inv,r6inv,r8inv,r10inv,ddexp,invexp;
  double t6,damp6,e6,tmp6,t8,damp8,e8,tmp8;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  error->warning(FLERR,"CN calculation...");
  calc_NCo();
  error->warning(FLERR,"Done");

  double **x = atom->x;
  double **f = atom->f;
  int *type  = atom->type;
  int nlocal = atom->nlocal;

  double *special_lj = force->special_lj;   
  int newton_pair = force->newton_pair;

  inum = list->inum;                
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) { // why not nlocal instead of inum?

    i = ilist[ii];
    itype = map[type[i]];

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = map[type[j]];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];

      rsq = delx*delx + dely*dely + delz*delz;
    
//      if (rsq < cutsq[itype][jtype]) {
      if (rsq < cutoff*cutoff) {

        r = sqrt(rsq);
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        r8inv = r2inv*r2inv*r2inv*r2inv;
        r10inv = r2inv*r2inv*r2inv*r2inv*r2inv;

        rr = r/r0ab[itype][jtype];

        error->warning(FLERR,"C6 calculation...");
        double C6 = getc6(itype,jtype,NCo[i],NCo[j]);
        double C8 = 3.0*C6*r2r4[itype]*r2r4[jtype];
        error->warning(FLERR,"Done");

        double alp6 = alpha;
        double alp8 = alpha+2;

        // we assume DFTD3 zero damping here
        t6=pow(rscale6/r,alp6);
        damp6 = 1.0/( 1.0+6.0*t6 );
        t8=pow(rscale8/r,alp8);
        damp8 =1.0/( 1.0+6.0*t8 );

        e6 = -scale6*C6*damp6*r6inv;
        e8 = -scale8*C8*damp8*r8inv;

        tmp6 = 6*scale6*C6*r8inv*damp6;
        tmp8 = 8*scale8*C8*r10inv*damp8;
        fpair = - tmp6 + (tmp6*alp6*t6*damp6) - tmp8 + (3/4*tmp8*alp8*t8*damp8);

        fpair *= factor_lj;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = e6 + e8;
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

void PairDFTD3::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  error->warning(FLERR,"Setting...");
  cutoff = utils::numeric(FLERR,arg[0],false,lmp);
  scale6 = utils::numeric(FLERR,arg[1],false,lmp);
  scale8 = utils::numeric(FLERR,arg[2],false,lmp);
  error->warning(FLERR,"Done");

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cutoff;
  }
  error->warning(FLERR,"ntypes :");
  error->warning(FLERR,std::to_string(atom->ntypes));

}

void PairDFTD3::coeff(int narg, char **arg)
{
  error->warning(FLERR,"Coefficients...");
  if (narg < 6 || narg > 7) error->all(FLERR, "Incorrect args for pair coefficients");

  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double r0ab_one = utils::numeric(FLERR,arg[2],false,lmp);
  rscale6 = utils::numeric(FLERR,arg[3],false,lmp);
  rscale8 = utils::numeric(FLERR,arg[4],false,lmp);
  alpha = utils::numeric(FLERR,arg[5],false,lmp);

  double cut_one = cutoff;
  if (narg == 4) cut_one = utils::numeric(FLERR, arg[6], false, lmp);

  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      r0ab[i][j] = r0ab_one;
      setflag[i][j] = 1;
    }
  }
  error->warning(FLERR,"Done");

}

/* ---------------------------------------------------------------------------------------------------------- */

void PairDFTD3::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style DFTD3 requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style DFTD3 requires newton pair on");

// need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  // create pages if first time or if neighbor pgsize/oneatom has changed
  int create = 0;
  if (ipage == nullptr) create = 1;
  if (pgsize != neighbor->pgsize) create = 1;
  if (oneatom != neighbor->oneatom) create = 1;

  if (create) {
    delete[] ipage;
    pgsize = neighbor->pgsize;
    oneatom = neighbor->oneatom;

    int nmypage = comm->nthreads;
    ipage = new MyPage<int>[nmypage];
    for (int i = 0; i < nmypage; i++)
      ipage[i].init(oneatom,pgsize);
  }
}

double PairDFTD3::init_one(int i, int j)
{
  error->warning(FLERR,"Initializing...");
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  cut[j][i] = cut[i][j] = cutoff;
  return cutoff;
  error->warning(FLERR,"Done...");

}
/* ---------------------------------------------------------------------------------------------------------- */

void PairDFTD3::allocate()
{
  error->warning(FLERR,"Allocating...");

  allocated = 1;
  int n = atom->ntypes;

  map = new int[n+1];

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(r0ab,n+1,n+1,"pair:r0ab");                   // not sure about this
  error->warning(FLERR,"Done");

}

void PairDFTD3::calc_NCo()
{
  int nj,*neighptrj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int inum,jnum,i,j,ii,jj,itype,jtype;
  double rr,rco,rsq,delrj[3];

  double **x = atom->x;
  int *type  = atom->type;

  if (atom->nmax > nmax) {

    nmax = atom->nmax;
    memory->grow(NCo,nmax,"pair:NCo");
    //memory->grow(bbij,nmax,MAXNEIGH,"pair:bbij");
  }


  inum = list->inum ;// !
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  ipage->reset();

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    nj = 0;
    neighptrj = ipage->vget();

    itype = map[type[i]];

    error->warning(FLERR,"itype : ");
    error->warning(FLERR,std::to_string(itype));
    error->warning(FLERR,std::to_string(type[i]));

    jlist = firstneigh[i];
    jnum = numneigh[i];

    NCo[i] = 0.0;

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj] & NEIGHMASK;
      jtype = map[type[j]];

      delrj[0] = x[i][0] - x[j][0];
      delrj[1] = x[i][1] - x[j][1];
      delrj[2] = x[i][2] - x[j][2];
    
      rsq = delrj[0]*delrj[0] + delrj[1]*delrj[1] + delrj[2]*delrj[2];

      error->warning(FLERR,"... before if ...");

      if (rsq > cn_thr) continue; //    warning HERE 

      neighptrj[nj++] = j;
  
      rr = sqrt(rsq);
      error->warning(FLERR,"... before rcov ...");
      rco = rcov[itype] + rcov[jtype];

      error->warning(FLERR,"... NCo calculation ...");
      NCo[i] += 1.0 / (1.0 + exp(K1 * ((rco/rr)-1.0)));
      error->warning(FLERR,"... NCo calculation done ...");

    }

    error->warning(FLERR,"...after if...");

    ipage->vgot(nj);

    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }

  // communicating coordination number to all nodes
  comm->forward_comm(this);
}

/* ---------------------------------------------------------------------------------------------------------- */

double PairDFTD3::getc6(int iat, int jat, double cni, double cnj){

  double c6_ref, itype, jtype, cni_ref, cnj_ref;
  double c6, c6mem, r, r_save, rsum, csum, tmp;

  c6 = 0; c6mem=-1.0E99, rsum = 0; csum = 0, r_save = 1.0E99;

  if (comm->me == 0) {

    PotentialFileReader reader(lmp, "../../potentials/pars.dtd3", "DFTD3", false);
    char *line;

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        c6_ref  = values.next_double();
        itype   = values.next_double();
        jtype   = values.next_double();
        cni_ref = values.next_double();
        cnj_ref = values.next_double();

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      if (itype == iat && jtype == jat){
        r = (cni - cni_ref) * (cni - cni_ref) + (cnj - cnj_ref) * (cnj - cnj_ref);

        if (r < r_save) {
          r_save = r;
          c6mem = c6_ref;
        }

        tmp = exp(K3*r); 
        rsum += tmp;
        csum += tmp * c6_ref;
      }
    }

    if (rsum > 1.0E-99) {
      c6 = csum / rsum;
    } else {
      c6 = c6mem;
    }
  }  
  return c6;
}

/* ---------------------------------------------------------------------------------------------------------- */

int PairDFTD3::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i ++) {
    j = list[i];
    buf[m++] = NCo[j]; 
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n ;
  for (i = first; i < last; i++) NCo[i] = buf[m++];
}

/* ---------------------------------------------------------------------------------------------------------- */

void PairDFTD3::write_restart(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++){
    fwrite(&NCo[i],sizeof(int),1,fp);
  }
}