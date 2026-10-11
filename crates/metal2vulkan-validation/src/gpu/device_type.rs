//! Explicit device selection.
//!
//! Nothing in this crate ranks devices. A caller names the kind of device it wants.

/// Kind of Vulkan physical device a caller asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceType {
    /// A CPU implementation (lavapipe, SwiftShader). Matches only
    /// `VK_PHYSICAL_DEVICE_TYPE_CPU`; when none exists the open fails, it never falls back.
    Cpu,
    /// A hardware or virtual GPU.
    Gpu,
}

impl DeviceType {
    /// Parse a `--device-type` value. There is no default: callers must name one.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(format!(
                "unknown device type {other:?}; expected \"cpu\" or \"gpu\""
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_type_has_no_default_spelling() {
        assert_eq!(DeviceType::parse("cpu").unwrap(), DeviceType::Cpu);
        assert_eq!(DeviceType::parse("gpu").unwrap(), DeviceType::Gpu);
        for value in ["", "CPU", "discrete", "integrated", "any"] {
            assert!(DeviceType::parse(value).is_err(), "{value:?}");
        }
        for device_type in [DeviceType::Cpu, DeviceType::Gpu] {
            assert_eq!(
                DeviceType::parse(device_type.as_str()).unwrap(),
                device_type
            );
        }
    }
}
