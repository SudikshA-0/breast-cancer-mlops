"""Request / response schemas."""
from pydantic import BaseModel, Field


class BreastCancerInput(BaseModel):
    radius_mean: float = Field(..., gt=0, examples=[17.99])
    texture_mean: float = Field(..., gt=0, examples=[10.38])
    perimeter_mean: float = Field(..., gt=0, examples=[122.8])
    area_mean: float = Field(..., gt=0, examples=[1001])
    smoothness_mean: float = Field(..., gt=0, examples=[0.1184])
    compactness_mean: float = Field(..., gt=0, examples=[0.2776])
    concavity_mean: float = Field(..., gt=0, examples=[0.3001])
    concave_points_mean: float = Field(..., alias="concave points_mean", gt=0, examples=[0.1471])
    symmetry_mean: float = Field(..., gt=0, examples=[0.2419])
    fractal_dimension_mean: float = Field(..., gt=0, examples=[0.07871])
    radius_se: float = Field(..., gt=0, examples=[1.095])
    texture_se: float = Field(..., gt=0, examples=[0.9053])
    perimeter_se: float = Field(..., gt=0, examples=[8.589])
    area_se: float = Field(..., gt=0, examples=[153.4])
    smoothness_se: float = Field(..., gt=0, examples=[0.006399])
    compactness_se: float = Field(..., gt=0, examples=[0.04904])
    concavity_se: float = Field(..., gt=0, examples=[0.05373])
    concave_points_se: float = Field(..., alias="concave points_se", gt=0, examples=[0.01587])
    symmetry_se: float = Field(..., gt=0, examples=[0.03003])
    fractal_dimension_se: float = Field(..., gt=0, examples=[0.006193])
    radius_worst: float = Field(..., gt=0, examples=[25.38])
    texture_worst: float = Field(..., gt=0, examples=[17.33])
    perimeter_worst: float = Field(..., gt=0, examples=[184.6])
    area_worst: float = Field(..., gt=0, examples=[2019])
    smoothness_worst: float = Field(..., gt=0, examples=[0.1622])
    compactness_worst: float = Field(..., gt=0, examples=[0.6656])
    concavity_worst: float = Field(..., gt=0, examples=[0.7119])
    concave_points_worst: float = Field(..., alias="concave points_worst", gt=0, examples=[0.2654])
    symmetry_worst: float = Field(..., gt=0, examples=[0.4601])
    fractal_dimension_worst: float = Field(..., gt=0, examples=[0.1189])


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
