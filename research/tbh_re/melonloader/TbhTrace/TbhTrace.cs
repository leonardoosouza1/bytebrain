// TbhTrace — mod MelonLoader (IL2CPP) que LOGA o fluxo de código do TaskbarHero.
// Lê UserData/tbh_hooks.txt (uma linha por método: Namespace.Classe::Metodo) e, via HarmonyX,
// põe um prefix em cada um que escreve no log quando o método é chamado. Só OBSERVA (não altera).
// Objetivo: entender o fluxo de código em runtime. NÃO burla anti-cheat nem economia online.
using System; using System.IO; using System.Reflection; using System.Collections.Generic;
using MelonLoader; using HarmonyLib;
[assembly: MelonInfo(typeof(TbhTrace.Core), "TbhTrace", "0.1", "leo")]
[assembly: MelonGame(null, null)]
namespace TbhTrace {
  public class Core : MelonMod {
    static int _hits;
    public override void OnInitializeMelon() {
      var path = Path.Combine(MelonEnvironment.UserDataDirectory, "tbh_hooks.txt");
      if (!File.Exists(path)) { LoggerInstance.Warning("tbh_hooks.txt não encontrado em UserData/"); return; }
      var h = new HarmonyLib.Harmony("tbh.trace");
      int ok=0, fail=0;
      foreach (var raw in File.ReadAllLines(path)) {
        var line = raw.Trim();
        if (line.Length==0 || line.StartsWith("//") || !line.Contains("::")) continue;
        var parts = line.Split(new[]{"::"}, StringSplitOptions.None);
        var type = ResolveType(parts[0]); if (type==null) { fail++; continue; }
        MethodInfo mi=null;
        foreach (var m in type.GetMethods(BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.Static))
          if (m.Name==parts[1]) { mi=m; break; }
        if (mi==null) { fail++; continue; }
        try {
          h.Patch(mi, prefix: new HarmonyMethod(typeof(Core).GetMethod(nameof(Log), BindingFlags.Static|BindingFlags.NonPublic)));
          ok++;
        } catch { fail++; }
      }
      LoggerInstance.Msg($"hooks ativos: {ok} · falharam: {fail} (nome ofuscado / genérico não resolvem)");
    }
    static Type ResolveType(string full) {
      foreach (var a in AppDomain.CurrentDomain.GetAssemblies()) {
        var t = a.GetType(full, false); if (t!=null) return t;
      }
      return null;
    }
    static void Log(MethodBase __originalMethod) {
      _hits++;
      if (_hits % 1 == 0)  // loga toda chamada; troque p/ amostrar se floodar
        MelonLogger.Msg($"[flow] {__originalMethod.DeclaringType?.Name}.{__originalMethod.Name}");
    }
  }
}
