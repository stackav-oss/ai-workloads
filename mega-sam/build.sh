#!/bin/bash
git submodule update --init --recursive
pixi install
sed -i 's/return caster\.operator typename make_caster<T>::template cast_op_type<T>();/return caster;/' .pixi/envs/default/lib/python3.10/site-packages/torch/include/pybind11/cast.h
sed -i -E -e "/^[[:space:]]*#?[[:space:]]*'-gencode=arch=compute_(60|61|70|75|80),code=sm_\\1',[[:space:]]*$/d" -e "/^[[:space:]]*'-O(2|3)',[[:space:]]*$/a\\                    '-gencode=arch=compute_90,code=sm_90'," mega-sam/base/setup.py
cd mega-sam/base && pixi run python setup.py install
