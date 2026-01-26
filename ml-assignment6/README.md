# Inlämningsuppgift 6 - Konvolutionellt neuralt nätverk

Konvolutionellt neuralt nätverk (CNN) med utbytbara stubklasser. När stubbarna ersatts med riktiga implementationer kan nätverket tränas att känna igen och prediktera siffrorna 0 och 1 utifrån enkla tvådimensionella bilder.

Nätverket består utav:
* Ett konvolutionellt lager, som bearbetar 4x4-bilder med en 2x2-kernel.
* Ett 2x2 maxpooling-lager, som samplar ned bilden till 2x2.
* Ett flatten-lager, som plattar bilden till en dimension (2x2 till 1x4).
* Ett dense-lager bestående av en nod och fyra vikter, som predikterar siffran på bilden.

Dense-lagret är redan implementerat, men de konvolutionella lagren måste implementeras innan nätverket fungerar som tänkt.

När samtliga lager har blivit implementerade bör utdatan se ut såhär:

```bash
--------------------------------------------------------------------------------
Input:
[[1.0, 1.0, 1.0, 1.0],
[1.0, 0.0, 0.0, 1.0],
[1.0, 0.0, 0.0, 1.0],
[1.0, 1.0, 1.0, 1.0]]

Prediction:
[0.0]

Input:
[[0.0, 1.0, 0.0, 0.0],
[0.0, 1.0, 0.0, 0.0],
[0.0, 1.0, 0.0, 0.0],
[0.0, 1.0, 0.0, 0.0]]

Prediction:
[1.0]
--------------------------------------------------------------------------------
```

## Implementationer
* En C++-implementation finns [här](./cpp/README.md).
* En Python-implementation finns [här](./python/README.md).
