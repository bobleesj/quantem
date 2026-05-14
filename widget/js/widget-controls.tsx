// Shared MUI style tokens + control building blocks for quantem widgets.
//
// Canonical source: Show2D. Other widgets (Show3D, Show3DVolume, Show4DSTEM)
// import from here so visual style stays consistent across the package.
//
// Order of controls per widget stays flexible — these are building blocks,
// not a layout.

export const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;

export const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};

export const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

export const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

export const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};

export const typographyLabel = {
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
};
