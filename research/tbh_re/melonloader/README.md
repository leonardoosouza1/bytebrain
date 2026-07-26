# TbhTrace — observar o FLUXO de código do TaskbarHero (MelonLoader / IL2CPP)

Loga em runtime quais métodos do jogo disparam. Só observa. Feito p/ Linux sem GPU (software render).

## 1. Instalar MelonLoader no jogo (Proton/Steam)
O TaskbarHero é Windows+IL2CPP rodando via Proton. MelonLoader roda dentro do prefixo Proton.
- Baixe MelonLoader (nightly IL2CPP) e extraia na pasta do jogo (onde está `TaskBarHero.exe`).
- 1ª execução: rode o jogo uma vez (via Steam). O MelonLoader gera os assemblies Il2Cpp interop
  em `MelonLoader/Il2CppAssemblies/` (é o que o mod referencia p/ resolver os tipos do jogo).
- Sem GPU: force render por software no comando de launch da Steam:
  `LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe %command%`
  (Unity 2D-ish roda em llvmpipe, lento mas serve p/ observar; se travar, tente `PROTON_USE_WINED3D=1`).

## 2. Compilar o mod
Precisa de .NET SDK (`dotnet`). No `TbhTrace/`:
- Referencie `MelonLoader.dll`, `0Harmony.dll` e os `Il2Cpp*` de `MelonLoader/Il2CppAssemblies/`.
- `dotnet build -c Release` → copie `TbhTrace.dll` p/ `<Jogo>/Mods/`.
(Um `TbhTrace.csproj` mínimo: TargetFramework net6.0, <Reference> pros DLLs acima.)

## 3. Escolher o que observar
- `python3 gen_hooks.py "save|account|progress" 40` → gera `tbh_hooks.txt`.
- Copie `tbh_hooks.txt` p/ `<Jogo>/UserData/`.

## 4. Rodar e ler o fluxo
- Rode o jogo. O log sai em `<Jogo>/MelonLoader/Latest.log` (linhas `[flow] Classe.Metodo`).
- Faça a ação no jogo (ex.: salvar) e veja a ordem em que os métodos disparam = o fluxo.

## 5. Laço autônomo (Claude testando)
Ciclo que o Claude toca sozinho, na SUA máquina (por isolamento):
1. Claude escolhe um sistema → `gen_hooks.py <sistema>` → `tbh_hooks.txt`.
2. `run_capture.sh` (você cria): `xvfb-run -a` + lança o jogo N segundos + mata + copia `Latest.log`.
3. Claude lê `Latest.log`, entende o fluxo, forma hipótese, refina os hooks → repete.
Assim o Claude mapeia o fluxo real sem você no meio — e tudo fica na sua máquina.

## Limite honesto
- MelonLoader IL2CPP no Linux/Proton é finicky (interop naming, software render). Pode precisar de ajuste.
- Métodos ofuscados / genéricos podem não resolver (o mod conta quantos falharam).
- Hook = observação. Alterar save/itens tem checagem no servidor + anti-cheat → não é o objetivo aqui.
