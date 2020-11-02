for file in *; do
   rm *.binvox
   if [[ $file == *.stl ]];then
        binvox -e $file
        
        binvox2bt "${file%.*}.binvox" -o  "${file%.*}.bt"
    fi
done
