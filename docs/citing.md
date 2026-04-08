---
icon: lucide/quote
---

# Citing ConfUSIus

If you use ConfUSIus in your research, please cite it using the following reference:

> Le Meur-Diebolt, S., & Cybis Pereira, F. (2026). ConfUSIus (v0.0.1-a23). Zenodo.
> https://doi.org/10.5281/zenodo.18611124

Or in BibTeX format:

```bibtex
@software{confusius,
  author    = {Le Meur-Diebolt, Samuel and Cybis Pereira, Felipe},
  title     = {ConfUSIus},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.0.1-a23},
  doi       = {10.5281/zenodo.18611124},
  url       = {https://doi.org/10.5281/zenodo.18611124}
}
```

---

## Citing Third-Party Projects

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

The [ConfUSIus GUI](../gui/overview.md) is built on top of [napari](https://napari.org),
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

### SimpleITK

The [`registration`][confusius.registration] module uses
[SimpleITK](https://simpleitk.org) for image registration and resampling. If you use the
registration features in your research, please also cite SimpleITK:

> Beare, R., Lowekamp, B., & Yaniv, Z. (2018). Image Segmentation, Registration and
> Characterization in R with SimpleITK. *Journal of Statistical Software*, 86(8), 1–35.
> https://doi.org/10.18637/jss.v086.i08

Or in BibTeX format:

```bibtex
@article{simpleitk,
  author  = {Beare, Richard and Lowekamp, Bradley and Yaniv, Ziv},
  title   = {Image Segmentation, Registration and Characterization in {R} with {SimpleITK}},
  journal = {Journal of Statistical Software},
  year    = {2018},
  volume  = {86},
  number  = {8},
  pages   = {1--35},
  doi     = {10.18637/jss.v086.i08},
  url     = {https://doi.org/10.18637/jss.v086.i08}
}
```
