// arch_flags, igemm_ukernel, igemm_size_fn, igemm_packed_offset_fn

#if XNN_ENABLE_KLEIDIAI
XNN_UKERNEL(xnn_arch_arm_sme, xnn_x16_pack_lh_ukernel__igemm_neonsme2,
            xnn_x16_pack_lh_size__igemm_neonsme2,
            xnn_x16_pack_lh_offset__igemm_neonsme2)
#endif  // XNN_ENABLE_KLEIDIAI
