import tenseal as ts
import utils

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

secret_context = context.serialize(save_secret_key = True)
utils.write_data("keys/secret.txt", secret_context)
  
context.make_context_public()
public_context = context.serialize()
utils.write_data("keys/public.txt", public_context)