---
hide:
    - navigation
icon: lucide/quote
---

# Citing ConfUSIus

If you use ConfUSIus in your research, please cite it using the following reference:

> Le Meur-Diebolt, S., & Cybis Pereira, F. (2026). ConfUSIus (v0.8.0). Zenodo.
> https://doi.org/10.5281/zenodo.18611124

Or in BibTeX format:

```bibtex
@software{confusius,
  author    = {Le Meur-Diebolt, Samuel and Cybis Pereira, Felipe},
  title     = {ConfUSIus},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.8.0},
  doi       = {10.5281/zenodo.18611124},
  url       = {https://doi.org/10.5281/zenodo.18611124}
}
```

---

## Third-Party Libraries

ConfUSIus stands on the shoulders of giants. It is built on top of many excellent
open-source projects, without which it could not exist. If you use the features listed
below, please consider citing the corresponding projects to support these efforts.

### BrainGlobe

The [`atlas`][confusius.atlas] module uses the [BrainGlobe Atlas
API](https://brainglobe.info) to interface with neuroanatomical atlases. If you use the
atlas features in your research, please also cite BrainGlobe:

> Claudi, F., Petrucco, L., Tyson, A. L., Branco, T., Margrie, T. W., & Portugues, R.
> (2020). BrainGlobe Atlas API: a common interface for neuroanatomical atlases.
> *Journal of Open Source Software*, 5(54), 2668.
> https://doi.org/10.21105/joss.02668

Or in BibTeX format:

```bibtex
@article{brainglobe,
  author    = {Claudi, Federico and Petrucco, Luigi and Tyson, Adam L. and
               Branco, Tiago and Margrie, Troy W. and Portugues, Ruben},
  title     = {{BrainGlobe} {Atlas} {API}: a common interface for neuroanatomical atlases},
  journal   = {Journal of Open Source Software},
  year      = {2020},
  volume    = {5},
  number    = {54},
  pages     = {2668},
  doi       = {10.21105/joss.02668},
  url       = {https://doi.org/10.21105/joss.02668}
}
```

### Napari

The [ConfUSIus GUI](../gui/overview) is built on top of [napari](https://napari.org),
a powerful multi-dimensional image viewer for Python. If you use the ConfUSIus GUI in
your research, please also cite napari:

> napari contributors (2019). napari: a multi-dimensional image viewer for
> Python. Zenodo. https://doi.org/10.5281/zenodo.3555620

Or in BibTeX format:

```bibtex
@software{napari,
  author    = {{napari contributors}},
  title     = {napari: a multi-dimensional image viewer for {Python}},
  year      = {2019},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.3555620},
  url       = {https://doi.org/10.5281/zenodo.3555620}
}
```

### Nilearn

The [`signal`][confusius.signal], [`glm`][confusius.glm], and
[`connectivity`][confusius.connectivity] modules contain code derived from
[Nilearn](https://nilearn.github.io). If you use these modules in your research, please
also cite Nilearn:

> Nilearn contributors (2023). Nilearn. Zenodo.
> https://doi.org/10.5281/zenodo.8397156

Or in BibTeX format:

```bibtex
@software{nilearn,
  author    = {{Nilearn contributors}},
  title     = {Nilearn},
  year      = {2023},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.8397156},
  url       = {https://doi.org/10.5281/zenodo.8397156}
}
```

### ITK

The [`registration`][confusius.registration] module uses
[ITK](https://itk.org) for image registration and resampling. If you use the
registration features in your research, please also cite ITK:

> Avants, B. B., Tustison, N. J., Stauffer, M., Song, G., Wu, B., & Gee, J. C. (2014).
> The Insight ToolKit image registration framework. *Frontiers in Neuroinformatics*, 8, 44.
> https://doi.org/10.3389/fninf.2014.00044

Or in BibTeX format:

```bibtex
@article{itk,
  author  = {Avants, Brian B. and Tustison, Nicholas J. and Stauffer, Michael and
             Song, Gang and Wu, Brian and Gee, James C.},
  title   = {The {Insight ToolKit} image registration framework},
  journal = {Frontiers in Neuroinformatics},
  year    = {2014},
  volume  = {8},
  pages   = {44},
  doi     = {10.3389/fninf.2014.00044},
  url     = {https://doi.org/10.3389/fninf.2014.00044}
}
```

## Datasets

### Nunez-Elizalde et al. (2022)

The [`fetch_nunez_elizalde_2022`][confusius.datasets.fetch_nunez_elizalde_2022]
function provides a fUSI-BIDS conversion of this dataset. If you use it, please cite:

> Nunez-Elizalde, A. O., Krumin, M., Reddy, C. B., Montaldo, G., Urban, A., Harris,
> K. D., & Carandini, M. (2022). Neural correlates of blood flow measured by
> ultrasound. *Neuron*, 110(10), 1631–1640.e4.
> https://doi.org/10.1016/j.neuron.2022.02.012

Or in BibTeX format:

```bibtex
@article{nunez-elizalde_neural_2022,
  title = {Neural correlates of blood flow measured by ultrasound},
  volume = {110},
  issn = {08966273},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0896627322001775},
  doi = {10.1016/j.neuron.2022.02.012},
  language = {en},
  number = {10},
  journal = {Neuron},
  author = {Nunez-Elizalde, Anwar O. and Krumin, Michael and Reddy, Charu Bai and
            Montaldo, Gabriel and Urban, Alan and Harris, Kenneth D. and Carandini, Matteo},
  month = may,
  year = {2022},
  pages = {1631--1640.e4},
}
```

License: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

### Cybis Pereira et al. (2026)

The [`fetch_cybis_pereira_2026`][confusius.datasets.fetch_cybis_pereira_2026]
function provides a fUSI-BIDS conversion of this dataset. If you use it, please cite:

> Cybis Pereira, F., Castedo, S. H., Meur-Diebolt, S. L., Ialy-Radio, N., Bhattacharya,
> S., Ferrier, J., Osmanski, B. F., Cocco, S., Monasson, R., Pezet, S., & Tanter, M.
> (2026). A vascular code for speed in the spatial navigation system. *Cell Reports*,
> 45(1). https://doi.org/10.1016/j.celrep.2025.116791

Or in BibTeX format:

```bibtex
@article{cybispereiraVascularCodeSpeed2026,
  title = {A Vascular Code for Speed in the Spatial Navigation System},
  author = {Cybis Pereira, Felipe and Castedo, Sebastian H. and {Meur-Diebolt}, Samuel Le and
            {Ialy-Radio}, Nathalie and Bhattacharya, Soumee and Ferrier, Jeremy and
            Osmanski, Bruno F{\'e}lix and Cocco, Simona and Monasson, Remi and
            Pezet, Sophie and Tanter, Micka{\"e}l},
  year = 2026,
  month = jan,
  journal = {Cell Reports},
  volume = {45},
  number = {1},
  publisher = {Elsevier},
  issn = {2211-1247},
  doi = {10.1016/j.celrep.2025.116791},
  urldate = {2025-12-30},
  langid = {english},
  keywords = {animal speed,cerebral blood volume,continuous attractor network,
              CP: neuroscience,freely moving,functional ultrasound imaging,
              hippocampus,locomotion,path integration,spatial navigation},
}
```

License: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

### Landemard et al. (2026)

The [`fetch_landemard_2026`][confusius.datasets.fetch_landemard_2026]
function provides a fUSI-BIDS re-export of this dataset. If you use it,
please cite:

> Landemard, A., Krumin, M., Harris, K. D., & Carandini, M. (2026). Brainwide
> blood volume reflects opposing neural populations. *Nature*.
> https://doi.org/10.1038/s41586-026-10350-9

Or in BibTeX format:

```bibtex
@article{landemard_brainwide_2026,
  title = {Brainwide blood volume reflects opposing neural populations},
  author = {Landemard, Agn{\`e}s and Krumin, Michael and Harris, Kenneth D. and
            Carandini, Matteo},
  year = {2026},
  month = apr,
  journal = {Nature},
  publisher = {Springer Science and Business Media {LLC}},
  doi = {10.1038/s41586-026-10350-9},
  url = {https://doi.org/10.1038/s41586-026-10350-9}
}
```

License: **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.

### Pereira et al. (2025)

The [`fetch_pereira_2025`][confusius.datasets.fetch_pereira_2025] function provides a
fUSI-BIDS re-export of this dataset. If you use it, please cite:

> Pereira, M., Droguerre, M., Valdebenito, M., Vidal, L., Marcy, G., Benkeder, S.,
> Marchal, P., Comte, J.-C., Pascual, O., Zimmer, L., & Vidal, B. (2025). Induction of
> haemodynamic travelling waves by glial-related vasomotion in a rat model of
> neuroinflammation: Implications for functional neuroimaging. *eBioMedicine*, 116,
> 105777. https://doi.org/10.1016/j.ebiom.2025.105777

```bibtex
@article{pereira_induction_2025,
  title = {Induction of haemodynamic travelling waves by glial-related vasomotion in a rat model of neuroinflammation: implications for functional neuroimaging},
  volume = {116},
  doi = {10.1016/j.ebiom.2025.105777},
  journal = {eBioMedicine},
  author = {Pereira, Mickaël and Droguerre, Marine and Valdebenito, Marco and Vidal, Louis and Marcy, Guillaume and Benkeder, Sarah and Marchal, Paul and Comte, Jean-Christophe and Pascual, Olivier and Zimmer, Luc and Vidal, Benjamin},
  year = {2025},
  pages = {105777},
}
```

License: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

### Pepe, Mariani et al. (2026)

The [`fetch_pepe_mariani_2026`][confusius.datasets.fetch_pepe_mariani_2026] function
provides a fUSI-BIDS re-export of this dataset. If you use it, please cite:

> Pepe, C., Mariani, J.-C., Urosevic, M., Gini, S., Stuefer, A., Ricci, F., Galbusera,
> A., Iurilli, G., & Gozzi, A. (2026). Structural and dynamic embedding of the mouse
> functional connectome revealed by functional ultrasound imaging (Fusi).
> https://doi.org/10.64898/2026.02.05.704055

```bibtex
@misc{pepe_structural_2026,
  title = {Structural and dynamic embedding of the mouse functional connectome revealed by functional ultrasound imaging ({Fusi})},
  doi = {10.64898/2026.02.05.704055},
  url = {http://biorxiv.org/lookup/doi/10.64898/2026.02.05.704055},
  author = {Pepe, Chiara and Mariani, Jean-Charles and Urosevic, Mila and Gini, Silvia and Stuefer, Alexia and Ricci, Fabio and Galbusera, Alberto and Iurilli, Giuliano and Gozzi, Alessandro},
  year = {2026},
}
```

License: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

### Khallaf et al. (2026)

The [`fetch_khallaf_2026`][confusius.datasets.fetch_khallaf_2026]
function provides a fUSI-BIDS conversion of this dataset. If you use it, please cite:

> Khallaf, M. A., Hart, D. W., Luo, W., Murad, F., Cybis Pereira, F., Mendez-Aranda, D.,
> Hagenah, N., Rossi, A., Bégay, V., Okrouhlík, J., Krautwurst, D., Ngalameno, M. K.,
> Ganswindt, A., Barker, A. J., Šumbera, R., Knaden, M., Pezet, S., Woehler, A.,
> Hansson, B. S., Bennett, N. C., & Lewin, G. R. (2026). A queen odour mediates
> reproductive suppression in a eusocial mammal. *Nature*.
> https://doi.org/10.1038/s41586-026-10772-5

Or in BibTeX format:

```bibtex
@article{khallafQueenOdourMediates2026,
  title = {A Queen Odour Mediates Reproductive Suppression in a Eusocial Mammal},
  author = {Khallaf, Mohammed A. and Hart, Daniel W. and Luo, Wenhan and
            Murad, Firdevs and Cybis Pereira, Felipe and {Mendez-Aranda}, Daniel and
            Hagenah, Nicole and Rossi, Alice and B{\'e}gay, Val{\'e}rie and
            Okrouhl{\'i}k, Jan and Krautwurst, Dietmar and Ngalameno, Mungo Kisinza and
            Ganswindt, Andre and Barker, Alison J. and {\v S}umbera, Radim and
            Knaden, Markus and Pezet, Sophie and Woehler, Andrew and
            Hansson, Bill S. and Bennett, Nigel C. and Lewin, Gary R.},
  year = 2026,
  month = jul,
  journal = {Nature},
  issn = {0028-0836, 1476-4687},
  doi = {10.1038/s41586-026-10772-5},
  url = {https://doi.org/10.1038/s41586-026-10772-5},
  urldate = {2026-07-15},
}
```

License: **[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)**.

## Templates

### Brunner, Grillet et al. (2021)

The [`fetch_template_brunner_grillet_2021`][confusius.datasets.fetch_template_brunner_grillet_2021]
function provides an Allen-aligned vascular mouse template. If you use this template
in your research, please cite the paper and source dataset:

> Brunner, C., Grillet, M., Urban, A., Roska, B., Montaldo, G., & Macé, E. (2021).
> Whole-brain functional ultrasound imaging in awake head-fixed mice.
> *Nature Protocols*, 16, 3547–3571.
> https://doi.org/10.1038/s41596-021-00548-8

> Macé, E., & Montaldo, G. (2020). Whole-brain visually evoked activity maps in
> head-fixed awake mice (June 2021 metadata update). *Zenodo*.
> https://doi.org/10.5281/zenodo.4905862

Or in BibTeX format:

```bibtex
@article{brunner2021,
  title = {Whole-brain functional ultrasound imaging in awake head-fixed mice},
  author = {Brunner, Clément and Grillet, Micheline and Urban, Alan and Roska, Botond and
            Montaldo, Gabriel and Macé, Emilie},
  journal = {Nature Protocols},
  volume = {16},
  number = {7},
  pages = {3547--3571},
  year = {2021},
  doi = {10.1038/s41596-021-00548-8},
}

@dataset{mace_montaldo2020,
  title = {Whole-brain visually evoked activity maps in head-fixed awake mice},
  author = {Macé, Emilie and Montaldo, Gabriel},
  publisher = {Zenodo},
  year = {2020},
  note = {June 2021 metadata update},
  doi = {10.5281/zenodo.4905862},
}
```

The template conversion reorders axes and assigns the Allen 50 µm coordinate grid
without changing vascular voxel values. Also cite Wang et al. (2020) for the Allen
CCF reference framework: <https://doi.org/10.1016/j.cell.2020.04.007>.

License: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

### Pepe, Mariani et al. (2026)

The [`fetch_template_pepe_mariani_2026`][confusius.datasets.fetch_template_pepe_mariani_2026]
function provides a mouse fUSI template derived from Pepe, Mariani et al. (2026). If you use
this template in your research, please also cite the corresponding article:

> Pepe, C., Mariani, J.-C., Urosevic, M., Gini, S., Stuefer, A., Ricci, F.,
> Galbusera, A., Iurilli, G., & Gozzi, A. (2026). Structural and dynamic embedding
> of the mouse functional connectome revealed by functional ultrasound imaging
> (fUSI). *bioRxiv*.
> https://doi.org/10.64898/2026.02.05.704055

Or in BibTeX format:

```bibtex
@article{pepe2026fusi,
  author    = {Pepe, Chiara and Mariani, Jean-Charles and Urosevic, Mila and Gini, Silvia and
               Stuefer, Alexia and Ricci, Fabio and Galbusera, Alberto and Iurilli, Giuliano and
               Gozzi, Alessandro},
  title     = {Structural and dynamic embedding of the mouse functional connectome revealed by
               functional ultrasound imaging ({fUSI})},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.02.05.704055},
  url       = {https://doi.org/10.64898/2026.02.05.704055}
}
```

License: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
