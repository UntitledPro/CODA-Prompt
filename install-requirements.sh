sewhile read requirement
    do conda install --yes $requirement || pip install $requirement
done < requirements.txt
conda install -c conda-forge pytorch-gpu