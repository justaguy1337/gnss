import { 
  BarChart3, 
  Upload, 
  Brain, 
  GitMerge,
  TrendingUp, 
  BookOpen, 
  Moon, 
  Sun,
  Satellite,
  Globe
} from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useDarkMode } from '../contexts/DarkModeContext';
import './Sidebar.css';

const Sidebar = () => {
  const { isDarkMode, toggleDarkMode } = useDarkMode();

  const navItems = [
    { to: '/', icon: BarChart3, label: 'Dashboard' },
    { to: '/upload', icon: Upload, label: 'Data Upload' },
    { to: '/insights', icon: Brain, label: 'Model Insights' },
    { to: '/meta-learner', icon: GitMerge, label: 'Meta Learner' },
    { to: '/earth', icon: Globe, label: 'Earth Visualization' },
    { to: '/impact', icon: TrendingUp, label: 'Impact & Benefits' },
    { to: '/research', icon: BookOpen, label: 'Research' },
  ];

  return (
    <div className="sidebar">
      {/* Logo and Brand */}
      <div className="sidebar-header">
        <div className="logo-container">
          <div className="logo-icon">
            <Satellite size={24} />
          </div>
          <div className="logo-text">
            <h1 className="logo-title">GNSS ErrorNet</h1>
            <p className="logo-subtitle">Satellite Error Prediction</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <item.icon size={20} />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Dark Mode Toggle */}
      <div className="sidebar-footer">
        <button
          onClick={toggleDarkMode}
          className="nav-link"
        >
          {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          <span>{isDarkMode ? 'Light Mode' : 'Dark Mode'}</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;