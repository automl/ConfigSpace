language: PYTHON
name: "HPOlib.cv"
variable {
  name: "preproc"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
  options: "2"
}
variable {
  name: "LOG_colnorm_thresh"
  size: 1
  type: FLOAT
  min: -20.7232658369
  max: -6.90775527898
}
variable {
  name: "pca_energy"
  size: 1
  type: FLOAT
  min: 0.5
  max: 1.0
}
variable {
  name: "Q16_LOG_nhid1"
  size: 1
  type: FLOAT
  min: 2.7726
  max: 6.9314718056
}
variable {
  name: "dist1"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
}
variable {
  name: "scale_heur1"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
}
variable {
  name: "scale_mult1"
  size: 1
  type: FLOAT
  min: 0.2
  max: 2.0
}
variable {
  name: "squash"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
}
variable {
  name: "iseed"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
  options: "2"
  options: "3"
}
variable {
  name: "batch_size"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
}
variable {
  name: "LOG_lr"
  size: 1
  type: FLOAT
  min: -13.6052
  max: 4.3948
}
variable {
  name: "Q1_LOG_lr_anneal_start"
  size: 1
  type: FLOAT
  min: 4.6051
  max: 9.2103
}
variable {
  name: "l2_penalty"
  size: 1
  type: ENUM
  options: "0"
  options: "1"
}
variable {
  name: "LOG_l2_penalty_nz"
  size: 1
  type: FLOAT
  min: -19.815510558
  max: -7.81551055796
}