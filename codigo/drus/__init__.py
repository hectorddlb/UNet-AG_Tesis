"""Réplica Python de DRUS (Zhang et al. 2023, DDRM para reconstrucción US).

Port de los scripts MATLAB del repo oficial
https://github.com/Yuxin-Zhang-Jasmine/DRUS-v1 bajo las convenciones del
pipeline unet_ag. La API pública está en ``drus.inference`` (clase ``DRUS``) y
la orden de línea de órdenes en ``drus.cli`` (``drus``); los artefactos que
necesita se construyen con ``drus.preparacion``.
"""
