function Patch = PatchGenerate(Seed,VertConn,VertArea,AreaDef)

Patch = Seed;
Area = sum(VertArea(Patch));
while Area <= AreaDef
    if Area <= AreaDef
        newverts = tess_scout_swell(Patch, VertConn);
        if ~isempty(newverts)
        Nouter = union(Patch,newverts);
        else
            break;
        end
        Area = sum(VertArea(Nouter));
    end
    if Area > AreaDef
        Ndiff = setdiff(Nouter,Patch);
        for i = 1: numel(Ndiff)
             Patch = union(Patch,Ndiff(i));
              Area = sum(VertArea(Patch));
            if Area > AreaDef
                break;
             end
        end
    else
         Patch = Nouter;
    end  
end
