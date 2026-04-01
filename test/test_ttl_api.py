"""Check available ttl API."""
import ttl
print("ttl attrs:", [x for x in dir(ttl) if not x.startswith('_')])
print()
print("Has 'core':", hasattr(ttl, 'core'))
print("Has 'node':", hasattr(ttl, 'node'))
print("Has 'grid_size':", hasattr(ttl, 'grid_size'))
print("Has 'kernel':", hasattr(ttl, 'kernel'))
print("Has 'operation':", hasattr(ttl, 'operation'))
