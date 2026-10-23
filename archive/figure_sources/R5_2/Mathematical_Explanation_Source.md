# Corrected CSSA and original CMGF interpretation

Equation(1):
```latex
U=\mathcal P_{\mathrm{in}}(F),\quad L=\mathcal L(U),\quad
C=\mathcal M([\operatorname{RowDWConv1d}_7(U),\operatorname{ColDWConv1d}_7(U)]),
A=\sigma(\mathcal A(C+L)),\quad Q=\sigma(\mathcal G([L,C,A])),
Z=Q\odot C+(1-Q)\odot L,\quad F'=F+\gamma\mathcal P_{\mathrm{out}}(Z).
```

New Equation(3a), original B2/B6 at fixed evaluation weights:
```latex
\Delta\ell = W[e_x\odot(g_s-g_{s_0})]+W(h_s-h_{s_0}).
```

Supplement algebra:
```latex
\ell(x,s)=We_x+W(e_x\odot g_s)+Wh_s+b,
\ell(x,s_0)=W_0e_x+b_0,\quad W_0=W\operatorname{diag}(1+g_{s_0}),\quad b_0=Wh_{s_0}+b.
```
These are editable manuscript equations. The decomposition is of logits, not probabilities, and is not a new experiment. The source network has not been modified.
