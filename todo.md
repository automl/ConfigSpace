* `normalized_default_value`, maybe include back in
* test default value of 0.5 w.r.t to log saling
* See about merging samplers for uniform float and int
* Test neighbors uniform integers includes both bounds
* Ordinal neighbors definition document as issue
* Techincally `legal` of uniform int was wrong due to just checking bounds, not rounding...
* Warnings for `_lower` and `_upper`
* Check in distribution that lower and upper are `(0, stepsize)`
* Allow for `None` in `legal_value`
* Remove T201 from pyproject ruff linting
* Why do we only validate configs when constructed with `values` and not from `vector`?
* Allow more types in constant
* why the restriction for pdf being 1d?
* Test `test_constant__pdf` was incorrect and it was passing non-normalized values to `_pdf`
and expecting the same result as if it was using `pdf`.
* Can we remove type checking rasing an error in `is_legal_vector`?

```python
c1 = NormalFloatHyperparameter("param", lower=0, upper=10, mu=3, sigma=2)
c2 = NormalFloatHyperparameter(
    "logparam",
    lower=np.exp(0),
    upper=np.exp(10),
    mu=3,     # <--------- for pdf to be same, shouldn't this be np.exp(3)
    sigma=2,  # <--------- and likewise, np.exp(2)?
    # What ends up happening is that this mu and sigma is applied to the `truncnorm` **after** the
    # log transform of upper and lower, i.e. the mu and sigma in `c2` and `c1` are applied in exactly the same
    # way. This seems less intuitive than just having it be applied in the range that was specified.
    log=True,
)

# Test that the pdf is the same for both
point_1 = np.array([3])
point_2_log = np.array([np.exp(3)])

# Give same pdf  (This seems wrong?)
assert c1.pdf(point_1)[0] == pytest.approx(0.2138045617479014)
assert c2.pdf(point_2_log)[0] == pytest.approx(0.2138045617479014)
```
* Convert range scalar back to unit scaler
* Bounding in transformations:
    * Value -> Vector  | No gaurantee pulled into vector bounds
    * Vector -> Value  | Gaurantee pulled into value bounds?


* pdf is only defined over vectorized range.
* To replace old q behaviour, introduce --> discretize which converts it to an ordinal.
* Finsih up with the fact quantization and log scaling are being annoying with each other.
* Fixed bug with pdf of oob integers for interger hyperparameters
* Default value of normal int correctly as integer value
* Mu and sigma for normal distributions are now always floats
* When testing normalized defaults, make sure that boundaries are set to `vectorized_lower` and `vectorized_upper`.
* Normalized values for integers are much closer to what would be expected, i.e. 0.5 being the halfway point.
* Allow for `None` in `Categorical`
* repr of hyperparameters were changed to be their `str()` output, `repr` is instead TODO
* `mu` and `sigma` are now always converted to float for both `NormalInteger` and `NormalFloatHyperparameter`
* Test neighborhood correctness
* `get_one_exchange_neighbourhood` seems to give strings of repeating numbers
* test `get_num_neighbors`
* Test neighborhood shuffles when n choices less than neighbors requested
* Star kwargs hyperparameters
* Serialization of UnParametrizedHyperparameter might be needed as it's got a different type
* Test that all inherited hyperparameters have `serialized_type_name`
* Can add multiple conditions at once...
* pcs (old) doesn't correctly read forbiddens with numericals:
    * They instead read as strings when it should be numbers

