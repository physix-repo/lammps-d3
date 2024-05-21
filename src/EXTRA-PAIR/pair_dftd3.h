/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
  -------------------------------------------------------------------------
  Contributed by
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(dftd3,PairDFTD3);
// clang-format on
#else

#ifndef LMP_PAIR_DFTD3_H
#define LMP_PAIR_DFTD3_H

#include "pair.h"

namespace LAMMPS_NS {

class PairDFTD3 : public Pair {

 public:

  PairDFTD3(class LAMMPS *);
  ~PairDFTD3() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;


  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  void write_restart(FILE *) override;

 protected:

  double cutoff, scale6, scale8;
  double **cut,**r0ab, rscale6, rscale8, alpha;

  int nmax;
  double *NCo;
  
  int pgsize, oneatom;
  MyPage<int> *ipage;

  void allocate();
  void calc_NCo();
  double getc6(int, int, double, double);
  double* getdc6(int, int, double, double);

};

}    // namespace LAMMPS_NS

#endif
#endif